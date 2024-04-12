from collections.abc import Iterable
from pathlib import Path
import pprint

from env_var import SOURCE_DIRECTORY

start_dir = SOURCE_DIRECTORY
by_extensions: dict[str, list[Path]] = {}


def walk(path: Path) -> Iterable[Path]:
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p


for p in walk(start_dir):
    ext = "".join(p.suffixes)
    if ext in by_extensions:
        by_extensions[ext].append(p)
    else:
        by_extensions[ext] = [p]

pprint.pp(by_extensions)
print("Extensions:", by_extensions.keys())
