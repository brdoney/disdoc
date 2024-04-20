from enum import Enum, auto
import re
from typing import Any  # type: ignore[reportAny]

from env_var import SOURCE_DIRECTORY
from typing_extensions import Self, override


class DocGroup(Enum):
    all = auto()
    ex0 = auto()
    ex1 = auto()
    ex2 = auto()
    ex3 = auto()
    ex4 = auto()
    ex5 = auto()
    p1 = auto()
    p2 = auto()
    p3 = auto()
    p4 = auto()
    midterm = auto()
    """Past midterms"""
    final = auto()
    """Past final exams"""
    admin = auto()
    """Adminstrative materials, including the syllabus and course policies"""
    material = auto()
    """Course materials, including lectures, example code, articles, and FAQs"""

    @override
    def __str__(self) -> str:
        """Return the name of the member directly."""
        return f"{self.name}"

    @classmethod
    def check_members(cls: type[Self]) -> None:
        exp_groups: set[str] = set()
        for group_dir in SOURCE_DIRECTORY.iterdir():
            group = group_dir.relative_to(SOURCE_DIRECTORY).parts[0]
            exp_groups.add(group)
        exp_groups.add("all")

        found_groups = set(str(member) for member in cls)

        assert (
            exp_groups == found_groups
        ), f"{cls} needs to be updated. Found members {found_groups}, but expected {exp_groups}."

    def get_filter(self) -> dict[str, Any] | None:
        """Gets the Chroma filter to use for a particular group."""
        if self is DocGroup.all:
            return None
        if self in EXERCISES or self in PROJECTS:
            # Add in material and admin to every assignment related query
            return {
                "group": {
                    "$in": [str(self), str(DocGroup.material), str(DocGroup.admin)]
                }
            }
        else:
            return {"group": str(self)}

    @staticmethod
    def from_str(s: str) -> "DocGroup":
        if s not in DocGroup.__members__:
            raise ValueError(f"Invalid category {s}")
        return DocGroup[s.lower()]


EXERCISES: set[DocGroup] = {g for g in DocGroup if re.match(r"ex\d", str(g))}
PROJECTS: set[DocGroup] = {g for g in DocGroup if re.match(r"p\d", str(g))}

# Check that our members are up to date on import
DocGroup.check_members()
