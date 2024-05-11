from copy import deepcopy
from pathlib import Path
import torch

from guided_transfer_learning import GuidanceMatrix

SCOUT_FILE_PREFIX = "gtl_scout"
DEFAULT_RAW_GUIDANCE_PATH = "raw_guidance_matrix.pt"
DEFAULT_MODEL_DIR = "."


class Scouts:
    def __init__(
        self,
        base_model,
        path=DEFAULT_MODEL_DIR,
        should_save_guidance=True,
        should_save_scouts=False,
        use_squared_differences=True,
    ):
        self.path = Path(path) if path else Path.cwd()
        self.should_save_guidance = should_save_guidance
        self.should_save_scouts = should_save_scouts
        self.use_squared_differences = use_squared_differences
        self.__scout_count = 0
        self.__base_model_params = self.__move_to_cpu(base_model)
        self.__sum = self.__initialize_sum()

    def __len__(self):
        return self.__scout_count

    def add_scout(self, scout_model):
        if self.should_save_scouts:
            self.__save_model(scout_model)
        scout_model = self.__move_to_cpu(scout_model)

        for layer in self.__sum:
            difference = torch.sub(scout_model[layer], self.__base_model_params[layer])
            self.__sum[layer] += self.__calculate_update_value(difference)
        self.__scout_count += 1

    def create_raw_guidance(
        self, device=None, path=DEFAULT_MODEL_DIR
    ) -> GuidanceMatrix:
        for layer in self.__sum:
            # self.__sum[layer] /= self.__scout_count
            self.__sum[layer] = self.__sum[layer].float() / self.__scout_count
        if self.should_save_guidance:
            path = Path(path) if path else self.path
            torch.save(self.__sum, path / DEFAULT_RAW_GUIDANCE_PATH)
        return (
            {k: self.__sum[k].to(device) for k in self.__sum} if device else self.__sum
        )

    def __calculate_update_value(self, difference):
        if self.use_squared_differences:
            return torch.square(difference)
        else:
            return torch.abs(difference)

    def __save_model(self, model):
        path = self.path / f"{SCOUT_FILE_PREFIX}_{self.__scout_count - 1}.pt"
        torch.save(model.state_dict(), path)

    @staticmethod
    def __move_to_cpu(model):
        return {k: v.cpu() for k, v in model.state_dict().items()}

    def __initialize_sum(self):
        sum_model = deepcopy(self.__base_model_params)
        for layer in sum_model:
            sum_model[layer] *= 0
        return sum_model
