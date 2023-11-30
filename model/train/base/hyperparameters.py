import enum
from ctypes import c_byte
from sys import getsizeof


class Hyperparameters:

    def __init__(self, hyperparams: enum.Enum):
        self.hyperparameters = hyperparams
        super().__init__()

    def set_values_from_dict(self, my_dict: dict) -> None:
        names = [hparams.name for hparams in self.hyperparameters]
        for k, v in my_dict.items():
            if k in names:
                Hyperparameters.change_value(self.hyperparameters[k].value, v)

    def get_values_as_dict(self) -> dict:
        names = [hparams.name for hparams in self.hyperparameters]
        result = dict()
        for k in names:
            result[k] = self.hyperparameters[k].value
        return result

    @staticmethod
    def get_values(obj) -> list:
        if isinstance(obj, Hyperparameters):
            return obj.hyperparameters.value

    @staticmethod
    def change_value(old_value: object, new_value: object) -> None:
        """
        Assigns contents of new object to old object.
        The size of new and old objection should be identical.

        Args:
            old_value (Any): Any object
            new_value (Any): Any object
        Raises:
            ValueError: Size of objects don't match
        Faults:
            Segfault: OOB write on destination
        """
        src_s, des_s = getsizeof(new_value), getsizeof(old_value)
        if src_s != des_s:
            raise ValueError("Size of new and old objects don't match")
        src_arr = (c_byte * src_s).from_address(id(new_value))
        des_arr = (c_byte * des_s).from_address(id(old_value))
        for index in range(len(des_arr)):
            des_arr[index] = src_arr[index]

    @staticmethod
    def from_enum_to_dict(hyperparams: enum.Enum) -> dict:
        names = [hparams.name for hparams in hyperparams]
        result = dict()
        for k in names:
            result[k] = hyperparams[k].value
        return result

    @staticmethod
    def from_dict_to_enum(name: str, values: dict):
        _k = _v = None

        class TheEnum(enum.Enum):
            nonlocal _k, _v
            for _k, _v in values.items():
                locals()[_k] = _v

        TheEnum.__name__ = name
        return TheEnum
