from enum import Enum, auto
from pathlib import Path


class TestType(Enum):
    FULL = auto()
    BACKEND = auto()
    FRONTEND = auto()

    @staticmethod
    def sorted() -> list["TestType"]:
        return [TestType.FULL, TestType.BACKEND, TestType.FRONTEND]

    @staticmethod
    def from_file(p: Path) -> "TestType":
        return TestType[p.stem.upper()]
