import numpy as np
from abc import ABC, abstractmethod
from typing import Type


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidFi(Fi):
    @staticmethod
    def _normal_dist_density(x: float, mean: float, sd: float):
        prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        return prob_density

    def __call__(self, state):
        # Value of density function is multiplied by 2000, so that its highest possible value is around 300
        return 2000 * HumanoidFi._normal_dist_density(state[0], 1.4, 0.05)


class FiFactory:
    FI_MAPPING = {
        'sum': SumFi,
        'humanoid': HumanoidFi,
        'default': HumanoidFi
    }

    @staticmethod
    def get_fi(name: str):
        fi = FiFactory.FI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {FiFactory.FI_MAPPING.keys()}")

        return fi()

    @staticmethod
    def register(name: str, _class=Type[Fi]):
        assert issubclass(_class, Fi), "Can only register classes that are subclasses of Fi"
        assert name not in FiFactory.FI_MAPPING, "This name is already taken"

        FiFactory.FI_MAPPING[name] = _class
