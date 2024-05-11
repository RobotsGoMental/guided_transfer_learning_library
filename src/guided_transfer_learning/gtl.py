from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt
import torch
from guided_transfer_learning import (
    AdjustGuidanceMatrix,
    Scouts,
    Temp,
    Focus,
    GuidanceMatrix,
)

MODELS_DIR = "."


def list_params(model: torch.nn.Module) -> List[tuple[str, torch.Size]]:
    """
    Returns a list of parameter names and their sizes for a given model.

    Args:
        model (torch.nn.Module): The PyTorch model for which to list the parameters.

    Returns:
        List[tuple[str, torch.Size]]): A list of tuples containing the parameter name and its size.

    Examples:
        >>> model = MyModel()
        >>> print(list_params(model))
        [('weight', torch.Size([64, 3, 5, 5])), ('bias', torch.Size([64]))]
    """
    state_dict = model.state_dict()
    return [(param_name, state_dict[param_name].size()) for param_name in state_dict]


def get_param_values(model: torch.nn.Module, first_n: int = 5):
    """
    Get the first n characters of each parameter in a given model.

    Args:
        model (torch.nn.Module): The model object to extract parameter values from.
        first_n (int, optional): The number of characters to extract from each parameter value. Defaults to 5.

    Returns:
        list: A list of the first n characters of each parameter value.

    """
    return [param[:first_n] for name, param in model.named_parameters()]


def create_scout_data_from_ranged_indexes(
    indexes: List[List[int]], features_or_labels: torch.Tensor
) -> List[torch.Tensor]:
    """Creates scout data from ranged indexes.

    Args:
        indexes (List[List[int]]): The ranged indexes.
        features_or_labels (torch.Tensor): The input data.

    Returns:
        List[torch.Tensor]: The scout data.
    """
    return _create_scout_data(expand_scout_indexes(indexes), features_or_labels)


def get_guidance_values(
    guidance_matrix: GuidanceMatrix, first_n=2
) -> dict[str, torch.Tensor]:
    """Return a dictionary containing the keys and the first 'first_n' values from the guidance matrix.

    Args:
        guidance_matrix (GuidanceMatrix): A dictionary containing the guidance values. The keys are identifiers and the values are lists.
        first_n (int): The number of values to retrieve from each list. Defaults to 2.

    Returns:
        dict: A dictionary containing the keys and the first 'first_n' values from the guidance matrix.

    """

    def slice_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor[:first_n]
        else:
            return torch.stack([slice_tensor(tensor[i]) for i in range(tensor.size(0))])

    return {key: slice_tensor(guidance_matrix[key]) for key in guidance_matrix}


def apply_guidance(model: torch.nn.Module, guidance_matrix: GuidanceMatrix) -> None:
    """
    Multiply the gradient of each parameter in the model with the corresponding value in the guidance matrix.

    Args:
        model (torch.nn.Module): The model whose parameters' gradients need to be adjusted.
        guidance_matrix (Guide): A dictionary containing the guidance values for each parameter in the model.

    Notes:
    - The guidance_matrix should have the same keys as the model's named_parameters.
    - The guidance_matrix values should be of the same shape as the corresponding parameter's gradient.
    - The gradient of each parameter will be multiplied element-wise with the corresponding value in the guidance_matrix.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad *= guidance_matrix[name]


def plot_guidance_distribution(
    guidance_matrix: GuidanceMatrix, name: str = None
) -> None:
    """
    Plots the guidance data from the guide matrix.

    Args:
        guidance_matrix (Guide): A dictionary containing the guidance data. Each key represents a name and the
            corresponding value represents the guidance data associated with that name.
        name (optional, str): The name for which the guidance data needs plotting. If not provided, all names from the
            guidance matrix will be plotted.

    Returns:
        None

    Examples:
        Creating a guide matrix
        >>> guidance_matrix = {'Name1': [1, 2, 3], 'Name2': [4, 5, 6], 'Name3': [7, 8, 9]}

        Plotting all names from the guide_matrix
        >>> plot_guidance_distribution(guidance_matrix)

        Plotting only one name from the guide_matrix
        >>> plot_guidance_distribution(guidance_matrix, name='Name2')
    """

    def plot(matrix: torch.Tensor, name: str) -> None:
        distribution = torch.histogram(torch.flatten(matrix).cpu())
        plt.plot(distribution[1][:-1].numpy(), distribution[0].numpy(), color="red")
        plt.title(name)
        plt.xlabel("Guiding parameter value")
        plt.ylabel("Count")
        plt.show()

    names = guidance_matrix.keys() if name is None else [name]
    for current_name in names:
        plot(guidance_matrix[current_name], current_name)


def adjust_guidance_matrix(
    guidance_matrix: GuidanceMatrix,
    focus: Focus | str = Focus.ZERO_ENFORCED_AND_NORMALIZED,
    temperature: Temp | str = Temp.ROOM,
    slope=None,
    intercept=None,
    should_save_guidance=True,
) -> GuidanceMatrix:
    adjusted_matrix = AdjustGuidanceMatrix(
        guidance_matrix, focus, temperature, slope, intercept, should_save_guidance
    )
    return adjusted_matrix.apply_focus_and_temperature()


def scale_guidance_matrix(guidance_matrix: GuidanceMatrix, factor: float) -> GuidanceMatrix:
    for param in guidance_matrix:
        guidance_matrix[param] *= factor
    return guidance_matrix


def c(number: int) -> list[int]:
    """
    Returns a list of length one containing the passed number.

    Args:
        number: must be an integer

    Returns:
        list: a list of length one containing the input number

    """
    return [number]


def create_scouts(
    model: torch.nn.Module,
    path: str = MODELS_DIR,
    should_save_guidance: bool = True,
    should_save_scouts: bool = False,
    use_squared_differences: bool = True,
) -> Scouts:
    return Scouts(
        model, path, should_save_guidance, should_save_scouts, use_squared_differences
    )


def expand_scout_indexes(
    indexes: List[List[int] | List[List[int]]],
) -> List[List[int]]:
    """Expands each scout index into a list of individual index values within the provided range.

    Args:
        indexes: A list of scout indexes, where each index can either be an integer or a list of two integers representing a range.

    Returns:
        list: A list of lists, where each inner list contains either a range object or a list of individual index values.

    Examples:
        >>> expand_scout_indexes([[1, 2, 4], [[7], [9]], [3, 5, 6], [[5], [15], 23, 24, [28], [30]], [40, 45, 46]])
        [[1, 2, 4], [7, 8, 9], [3, 5, 6], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 24, 28, 29, 30], [40, 45, 46]]
    """
    scout_indexes = []
    for scout in indexes:
        expanded_scout = []
        start_of_range = None
        for index_range_or_val in scout:
            if isinstance(index_range_or_val, list):
                if start_of_range:
                    expanded_scout.extend(
                        range(start_of_range, index_range_or_val[0] + 1)
                    )
                    start_of_range = None
                else:
                    start_of_range = index_range_or_val[0]
            else:
                expanded_scout.extend([index_range_or_val])
        scout_indexes.append(expanded_scout)

    return scout_indexes


def _create_scout_data(
    scout_indexes: List[List[int]], features_or_labels: torch.Tensor
) -> List[torch.Tensor]:
    """Create scout data by stacking selected data elements based on scout indexes.

    Args:
        scout_indexes (List[List[int]]): A list of lists containing the indexes of the data elements to be selected for
            each scout.
        features_or_labels (torch.Tensor): A list of tensors representing the data elements.

    Returns:
        List[torch.Tensor]: A list of stacked tensors, where each tensor is created by stacking the selected data elements based on the scout indexes.
    """
    return [
        torch.stack([features_or_labels[i] for i in index]) for index in scout_indexes
    ]
