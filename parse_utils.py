from enum import Enum, auto


def remove_suffix(x: str) -> float:
    if x.endswith("ms"):
        return float(x[:-2])
    elif x.endswith("s"):
        return float(x[:-1]) * 1000
    else:
        return float(x)


class GraphType(Enum):
    ABSOLUTE = auto()
    RELATIVE = auto()
    PERCENTAGE = auto()
