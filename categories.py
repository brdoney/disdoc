from enum import Enum, auto
import re
from typing import Any  # type: ignore[reportAny]

from env_var import SOURCE_DIRECTORY
from typing_extensions import override


class DocGroup(Enum):
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
    # manpages = auto()

    @override
    def __str__(self) -> str:
        """Return the name of the member directly."""
        return f"{self.name}"

    @classmethod
    def check_members(cls) -> None:
        exp_groups: set[str] = set()
        for group_dir in SOURCE_DIRECTORY.iterdir():
            group = group_dir.relative_to(SOURCE_DIRECTORY).parts[0]
            exp_groups.add(group)

        found_groups = set(str(member) for member in cls)

        unexpected = found_groups.difference(exp_groups)
        not_found = exp_groups.difference(found_groups)

        error_msg = f"{cls} needs to be updated."
        if unexpected:
            error_msg += f" Found the following unexpected members: {unexpected}."
        if not_found:
            error_msg += f" Could not find the following expected members: {not_found}."

        assert exp_groups == found_groups, error_msg

    def get_filter(self) -> dict[str, Any]:
        """Gets the Chroma filter to use for a particular group."""
        # Add in material and admin to every assignment related query
        if self in EXERCISES or self in PROJECTS:
            return {
                "group": {
                    "$in": [str(self), str(DocGroup.material), str(DocGroup.admin)]
                }
            }
        else:
            return {"group": str(self)}


EXERCISES: set[DocGroup] = {g for g in DocGroup if re.match(r"ex\d", str(g))}
PROJECTS: set[DocGroup] = {g for g in DocGroup if re.match(r"p\d", str(g))}

# Check that our members are up to date on import
DocGroup.check_members()
