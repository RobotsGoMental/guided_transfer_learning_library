from __future__ import annotations

from enum import Enum
from typing import Union, Any

import torch

GuidanceMatrix = dict[str, Any]


class Temp(Enum):
    """
    Temperature determines how restrictive the matrix will be for training. Lower temperature
    means that more parameters will be restricted in training. Higher temperature means that more
    parameters are allowed to be trained.

    There are six levels of temperature that can be selected staring from the coldest "freezing"
    and ending with the warmest "evaporating":

    "freezing" - Only top 1% of parameters with the highest guidance values are allowed to change.
    "icy" - Only the raw guidance values above the mean are allowed to change. Their values scale between 0.0 and
        maximum.
    "chilly" - All guidance values are scaled between zero and maximum. This is a different way to enforce zero.
    "room" (temperature) - If room temperature is selected, parameter values are not changed. This is the default.
    "warm" - More freedom is given for parameter changes. The guiding values begin with the mean of the guidance values.
    "evaporating" - This is the least restricted form of guidance. Guiding matrix only lightly touches the learning
        process. The guiding begin with the fourth quantile of the guiding matrix.
    """

    FREEZING = "freezing"
    ICY = "icy"
    CHILLY = "chilly"
    ROOM = "room"
    STEP = "step"
    WARM = "warm"
    EVAPORATING = "evaporating"


class Focus(Enum):
    """The focus determines how the guidance matrix will be adjusted. There are four types of focus:

    "raw" - The guidance matrix will not be adjusted.
    "zero_enforced" - The guidance matrix will be adjusted to enforce zero.
    "zero_enforced_and_normalized" - The guidance matrix will be adjusted to enforce zero and normalized.
    "normalized" - The guidance matrix will be adjusted to be normalized.
    """

    RAW = "raw"
    ZERO_ENFORCED = "zero_enforced"
    ZERO_ENFORCED_AND_NORMALIZED = "zero_enforced_and_normalized"
    NORMALIZED = "normalized"


class AdjustGuidanceMatrix:
    """Class for adjusting the guidance matrix based on different types and temperatures.

    Args:
        guidance_matrix (GuidanceMatrix): The guidance matrix containing the data for the guidance system.
        focus (Focus | str): The type of focus to be applied. Default is Type.ZERO_ENFORCED_AND_NORMALIZED.
        temperature (Temp | str): The temperature to be used for adjustment. Default is Temp.ROOM.
        slope (float): The slope value for adjustment. If not provided, no adjustment will be applied.
        intercept (float): The intercept value for adjustment. If not provided, no adjustment will be applied.
        should_save_guidance (bool): Whether or not to save the adjusted guidance. Default is False.

    Example usage:
    ```python
    guide = GuidanceMatrix()
    adjuster = AdjustGuidanceMatrix(guide)
    adjuster.apply_focus_and_temperature()
    ```
    """

    def __init__(
        self,
        guidance_matrix: GuidanceMatrix,
        focus: Focus | str = Focus.ZERO_ENFORCED_AND_NORMALIZED,
        temperature: Temp | str = Temp.ROOM,
        slope: float = None,
        intercept: float = None,
        should_save_guidance: bool = False,
    ):
        self.guidance_matrix = guidance_matrix
        self.focus = self.__interpret_enum(Focus, focus)
        self.temperature = self.__interpret_enum(Temp, temperature)
        self.slope = slope
        self.intercept = intercept
        self.should_save_guidance = should_save_guidance
        self.temperature_mapping = {
            Temp.CHILLY: self.adjust_temperature_chilly,
            Temp.ROOM: self.adjust_temperature_room,
            Temp.ICY: self.adjust_temperature_icy,
            Temp.FREEZING: self.adjust_temperature_freezing,
            Temp.STEP: self.adjust_temperature_freezing,
            Temp.WARM: self.adjust_temperature_warm,
            Temp.EVAPORATING: self.adjust_temperature_evaporating,
        }
        self.focus_mapping = {
            Focus.RAW: self.raw,
            Focus.ZERO_ENFORCED: self.zero_enforced,
            Focus.ZERO_ENFORCED_AND_NORMALIZED: self.zero_enforced_and_normalized,
            Focus.NORMALIZED: self.normalized,
        }

    @staticmethod
    def __interpret_enum(
        enum_class: Focus | Temp, enum_key: str | Focus | Temp
    ) -> Focus | Temp:
        if isinstance(enum_key, str):
            try:
                return enum_class[enum_key.upper()]
            except KeyError:
                name = "focus" if enum_class == Focus else "temperature"
                raise ValueError(f"Unknown {name}: {enum_key}.")
        return enum_key

    def adjust_temperature_freezing(self, layer: str) -> None:
        """Only top 1% of parameters with the highest guidance values are allowed to change.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        threshold = self.percentile(self.guidance_matrix[layer], 99)
        self.guidance_matrix[layer] = torch.where(
            self.guidance_matrix[layer] < threshold, 0.0, 1.0
        )

    def adjust_temperature_icy(self, layer: str) -> None:
        """Only the raw guidance values above the mean are allowed to change. Their values scale between 0.0 and
        maximum.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        slope = 1
        intercept = (
            torch.amax(self.guidance_matrix[layer])
            - torch.amin(self.guidance_matrix[layer])
        ) / 2
        self.guidance_matrix[layer] = (
            self.guidance_matrix[layer] * slope - intercept
        ) * 2

    def adjust_temperature_chilly(self, layer: str) -> None:
        """All guidance values are scaled between zero and maximum. This is a different way to enforce zero.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        slope = 1
        intercept = torch.amin(self.guidance_matrix[layer])
        self.guidance_matrix[layer] = self.guidance_matrix[layer] * slope - intercept

    def adjust_temperature_room(self, layer: str) -> None:
        """If room temperature is selected, parameter values are not changed. This is the default.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        pass  # Do nothing

    def adjust_temperature_warm(self, layer: str) -> None:
        """
        More freedom is given for parameter changes. The guiding values begin with the mean of the guidance values.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        slope = 0.5
        intercept = (
            -(
                torch.amax(self.guidance_matrix[layer])
                - torch.amin(self.guidance_matrix[layer])
            )
            / 2
        )
        self.guidance_matrix[layer] = self.guidance_matrix[layer] * slope - intercept

    def adjust_temperature_evaporating(self, layer: str) -> None:
        """
        This is the least restricted form of guidance. Guiding matrix only lightly touches the learning process. The
        guiding begin with the fourth quantile of the guiding matrix.

        Args:
            layer (str): The name of the layer to adjust.

        Returns:
            None
        """
        slope = 0.25
        intercept = -(
            torch.amax(self.guidance_matrix[layer])
            - torch.amin(self.guidance_matrix[layer])
        )
        self.guidance_matrix[layer] = self.guidance_matrix[layer] * slope - intercept

    def apply_temperature(self) -> None:
        """
        Applies temperature adjustments to the guide layers.

        Raises:
            ValueError: If the given temperature is not defined in the temperature_mapping.

        Returns:
            None
        """
        for layer in self.guidance_matrix:
            if self.slope is None or self.intercept is None:
                self.temperature_mapping[self.temperature](layer)
            else:
                self.guidance_matrix[layer] = (
                    self.guidance_matrix[layer] * self.slope - self.intercept
                )
            self.guidance_matrix[layer] = torch.where(
                self.guidance_matrix[layer] < 0, 0, self.guidance_matrix[layer]
            )

    @staticmethod
    def percentile(t: torch.tensor, q: float) -> Union[int, float]:
        """Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to ``numpy.percentile(..., interpolation="nearest")``.

        Args:
            t (torch.tensor): Input tensor.
            q (float): Percentile to compute, which must be between 0 and 100 inclusive.

        Returns:
            Resulting value (scalar).

        Notes:
            ``kthvalue()`` works one-based, i.e. the first sorted value indeed corresponds to k=1, not k=0!
            Use float(q) instead of q directly, so that ``round()`` returns an integer, even if q is a np.float32.

        Credit: https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611
        """
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    @staticmethod
    def zero_to_max_normalized(sum_diff: torch.Tensor) -> torch.Tensor:
        min_ls = torch.min(sum_diff)
        max_ls = torch.max(sum_diff)
        spread = max_ls - min_ls
        if spread > 0:
            return (sum_diff - min_ls) / spread
        else:
            return sum_diff

    @staticmethod
    def normalized_values(sum_diff: torch.Tensor) -> torch.Tensor:
        max_ls = torch.max(sum_diff)
        if max_ls > 0:
            return sum_diff / max_ls
        else:
            return sum_diff

    def raw(self):
        pass  # Do nothing

    def zero_enforced(self) -> None:
        for layer in self.guidance_matrix:
            self.guidance_matrix[layer] -= torch.min(self.guidance_matrix[layer])

    def zero_enforced_and_normalized(self) -> None:
        for layer in self.guidance_matrix:
            self.guidance_matrix[layer] = self.zero_to_max_normalized(
                self.guidance_matrix[layer]
            )

    def normalized(self) -> None:
        for layer in self.guidance_matrix:
            self.guidance_matrix[layer] = self.normalized_values(
                self.guidance_matrix[layer]
            )

    def apply_focus(self) -> None:
        self.focus_mapping[self.focus]()

    def apply_focus_and_temperature(self) -> GuidanceMatrix:
        """Applies focus and temperature to the guidance matrix.

        This method applies focus and temperature to the guidance matrix by calling the `apply_focus` and
        `apply_temperature` methods. If the flag `should_save_guidance` is set to `True`, the guidance matrix is saved
        to the file 'guidance_matrix.pt' using the torch.save() function.

        Returns:
            The guidance matrix after applying focus and temperature.
        """
        self.apply_focus()
        self.apply_temperature()
        if self.should_save_guidance:
            torch.save(self.guidance_matrix, "guidance_matrix.pt")
        return self.guidance_matrix
