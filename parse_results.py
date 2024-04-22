import json
import re
from collections.abc import Iterable, Iterator
from enum import Enum, auto
from pathlib import Path
from typing import Literal
from typing_extensions import overload

import matplotlib.style
import pandas as pd

from test_types import TestType

matplotlib.style.use("seaborn-v0_8")

RESULTS = Path("./results")

latency_pat = re.compile(r"Latency\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)%")
rps_pat = re.compile(r"Req\/Sec\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)%")
dist_pat = re.compile(r"\d+%\s+(\S+m?s)")


def pathname(p: Path) -> str:
    return p.name


newest = sorted(RESULTS.iterdir(), key=pathname)[-1]


def remove_suffix(x: str) -> float:
    if x.endswith("ms"):
        return float(x[:-2])
    elif x.endswith("s"):
        return float(x[:-1]) * 1000
    else:
        return float(x)


def extract_row(
    row_match: re.Match[str], dist_match: Iterator[re.Match[str]] | None = None
) -> list[str | float]:
    vals: list[str | float] = []
    vals.extend(remove_suffix(x) for x in row_match.groups())

    if dist_match is not None:
        for m in dist_match:
            vals.extend(remove_suffix(x) for x in m.groups())

    return vals


@overload
def tt_prefix(s: str) -> list[str]: ...


@overload
def tt_prefix(s: list[str]) -> list[str]: ...


def tt_prefix(s: str | list[str]) -> list[str]:
    if isinstance(s, str):
        return [f"{tt.name.title()} {s}" for tt in TestType]
    else:
        els: list[str] = []
        for el in s:
            for tt in TestType:
                prefixed = f"{tt.name.title()} {el}"
                if prefixed in latency_rows or prefixed in request_rows:
                    els.append(prefixed)
        return els


class GraphType(Enum):
    ABSOLUTE = auto()
    RELATIVE = auto()
    PERCENTAGE = auto()


def plot_latency(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[tt_prefix(["Stdev (ms)", "+/- Stdev (%)"])] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df[tt_prefix("Avg (ms)")] -= 1
        df[tt_prefix("Avg (ms)")] *= 100

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = "%" if graph_type is GraphType.PERCENTAGE else "ms"

    # mapper = dict(zip(tt_prefix("Stdev (ms)"), tt_prefix("Avg (ms)")))
    # yerr = (
    #     df[tt_prefix("Stdev (ms)")].rename(columns=mapper)
    #     if graph_type is not GraphType.PERCENTAGE
    #     else None
    # )
    yerr = None

    ax = df.plot(
        use_index=True,
        y=tt_prefix("Avg (ms)"),
        yerr=yerr,  # type: ignore[reportArgumentType]
        kind="bar",
        title=f"Scheduler vs. Average Latency{suffix}",
        ylabel=f"Average Latency ({y_units}){suffix} (lower is better)",
        xlabel="Scheduler",
        # legend=False,
        rot=45,
    )
    fig = ax.get_figure()
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig.savefig(f"latency-{graph_type.name.lower()}.png", bbox_inches="tight", dpi=200)  # type: ignore[reportUnknownMemberType]


def plot_tail_latency(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[tt_prefix(["Stdev (ms)", "+/- Stdev (%)"])] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df[tt_prefix("99% (ms)")] -= 1
        df[tt_prefix("99% (ms)")] *= 100

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = "%" if graph_type is GraphType.PERCENTAGE else "ms"

    ax = df.plot(
        use_index=True,
        y=tt_prefix("99% (ms)"),
        kind="bar",
        title=f"Scheduler vs. 99% Tail Latency{suffix}",
        ylabel=f"99% Tail Latency ({y_units}){suffix} (lower is better)",
        xlabel="Scheduler",
        # legend=False,
        rot=45,
    )
    fig = ax.get_figure()
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig.savefig(  # type: ignore[reportUnknownMemberType]
        f"tail-latency-{graph_type.name.lower()}.png", bbox_inches="tight", dpi=200
    )


def plot_requests(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[tt_prefix(["Stdev", "+/- Stdev (%)"])] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df[tt_prefix("Avg")] -= 1
        df[tt_prefix("Avg")] *= 100

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = " (%)" if graph_type is GraphType.PERCENTAGE else ""

    # mapper = dict(zip(tt_prefix("Stdev"), tt_prefix("Avg")))
    # yerr = (
    #     df[tt_prefix("Stdev")].rename(columns=mapper)
    #     if graph_type is not GraphType.PERCENTAGE
    #     else None
    # )
    yerr = None

    ax = df.plot(
        use_index=True,
        y=tt_prefix("Avg"),
        yerr=yerr,  # type: ignore[reportArgumentType]
        kind="bar",
        title=f"Scheduler vs. Average Requests/Second{suffix}",
        ylabel=f"Average Requests/Second{y_units}{suffix} (higher is better)",
        xlabel="Scheduler",
        # legend=False,
        rot=45,
    )
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig = ax.get_figure()
    fig.savefig(f"requests{graph_type.name.lower()}.png", bbox_inches="tight", dpi=200)  # type: ignore[reportUnknownMemberType]


def add_to_dict(
    d: dict[str, list[str | float]], it: Iterable[tuple[str, str | float]]
) -> None:
    for key, val in it:
        d[key].append(val)


base_lat_col_names = [
    "Avg (ms)",
    "Stdev (ms)",
    "Max (ms)",
    "+/- Stdev (%)",
    "50% (ms)",
    "75% (ms)",
    "90% (ms)",
    "99% (ms)",
]
lat_full_col_names = [f"Full {x}" for x in base_lat_col_names]
lat_frontend_col_names = [f"Frontend {x}" for x in base_lat_col_names]
lat_backend_col_names = dict(
    avg="Backend Avg (ms)",
    max="Backend Max (ms)",
    std="Backend Stdev (ms)",
    p50="Backend 50% (ms)",
    p75="Backend 75% (ms)",
    p90="Backend 90% (ms)",
    p99="Backend 99% (ms)",  # TODO: Fix this key
)
lat_col_names = [
    "Scheduler",
    *lat_full_col_names,
    *lat_frontend_col_names,
    *lat_backend_col_names.values(),
]

base_req_col_names = [
    "Avg",
    "Stdev",
    "Max",
    "+/- Stdev (%)",
]
req_full_col_names = [f"Full {x}" for x in base_req_col_names]
req_frontend_col_names = [f"Frontend {x}" for x in base_req_col_names]
req_backend_col_names = dict(
    avg="Backend Avg",
    std="Backend Stdev",
    max="Backend Max",
)
req_col_names = [
    "Scheduler",
    *req_full_col_names,
    *req_frontend_col_names,
    *req_backend_col_names.values(),
]

latency_rows: dict[str, list[str | float]] = {col: [] for col in lat_col_names}
request_rows: dict[str, list[str | float]] = {col: [] for col in req_col_names}


def parse_wrk(tt: TestType, result_file: Path) -> None:
    if tt == TestType.FULL:
        lat_cols = lat_full_col_names
        req_cols = req_full_col_names
    elif tt == TestType.FRONTEND:
        lat_cols = lat_frontend_col_names
        req_cols = req_frontend_col_names
    else:
        raise ValueError(f"Invalid test type for wrk: {tt}")

    with result_file.open() as f:
        contents = f.read()
        lat_m = latency_pat.search(contents)
        req_m = rps_pat.search(contents)
        dist_m = dist_pat.finditer(contents)
        if lat_m is None or req_m is None:
            raise ValueError(f"Invalid results file {sched_dir}")

        add_to_dict(latency_rows, zip(lat_cols, extract_row(lat_m, dist_m)))

        add_to_dict(request_rows, zip(req_cols, extract_row(lat_m)))


def parse_json(tt: TestType, result_file: Path) -> None:
    assert tt is TestType.BACKEND
    with result_file.open() as f:
        res: dict[Literal["latency"] | Literal["rps"], dict[str, str]] = json.load(f)
        for key, val in res["latency"].items():
            latency_rows[lat_backend_col_names[key]].append(float(val))
        for key, val in res["rps"].items():
            request_rows[req_backend_col_names[key]].append(float(val))


latency_csv = newest / "latency.csv"
requests_csv = newest / "requests.csv"


if latency_csv.exists() and requests_csv.exists():
    # Parsed dataframes already exist, so re-use them
    latency = pd.read_csv(latency_csv, index_col="Scheduler")  # type: ignore[reportUnknownMemberType]
    requests = pd.read_csv(requests_csv, index_col="Scheduler")  # type: ignore[reportUnknownMemberType]
else:
    for sched_dir in newest.iterdir():
        sched = sched_dir.stem

        latency_rows[lat_col_names[0]].append(sched)
        request_rows[req_col_names[0]].append(sched)

        for result in sched_dir.iterdir():
            tt = TestType.from_file(result)
            if tt is TestType.BACKEND:
                parse_json(tt, result)
            else:
                parse_wrk(tt, result)
    # We had to parse stuff, so convert it to df
    latency = pd.DataFrame(latency_rows)
    latency = latency.set_index("Scheduler")  # type: ignore[reportUnknownMemberType]

    requests = pd.DataFrame(request_rows)
    requests = requests.set_index("Scheduler")  # type: ignore[reportUnknownMemberType]

    latency.to_csv(latency_csv)
    requests.to_csv(requests_csv)

REMOVE_CENTRAL = True
if REMOVE_CENTRAL:
    latency = latency.drop("scx_central")  # type: ignore[reportUnknownMemberType]
    requests = requests.drop("scx_central")  # type: ignore[reportUnknownMemberType]

if input("Generate graphs? [y/N] ").lower() == "y":
    plot_latency(latency, GraphType.RELATIVE)
    plot_latency(latency, GraphType.PERCENTAGE)
    plot_tail_latency(latency, GraphType.RELATIVE)
    plot_tail_latency(latency, GraphType.PERCENTAGE)
    plot_requests(requests, GraphType.RELATIVE)
    plot_requests(requests, GraphType.PERCENTAGE)

latency_mins = latency.idxmin()  # type: ignore[reportUnknownMemberType]
print("Latency minimums:")
print(latency_mins[tt_prefix(["Avg (ms)", "99% (ms)"])])  # type: ignore[reportUnknownMemberType]

request_mins = requests.idxmax()  # type: ignore[reportUnknownMemberType]
print("RPS maximums:")
print(request_mins[tt_prefix("Avg")])  # type: ignore[reportUnknownMemberType]
