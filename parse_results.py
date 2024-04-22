import re
from collections.abc import Iterator
from enum import Enum, auto
from pathlib import Path

import matplotlib.style
import pandas as pd

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
    sched: str,
    row_match: re.Match[str],
    dist_match: Iterator[re.Match[str]] | None = None,
) -> list[str | float]:
    vals: list[str | float] = [sched]
    vals.extend(remove_suffix(x) for x in row_match.groups())

    if dist_match is not None:
        for m in dist_match:
            vals.extend(remove_suffix(x) for x in m.groups())

    return vals


class GraphType(Enum):
    ABSOLUTE = auto()
    RELATIVE = auto()
    PERCENTAGE = auto()


def plot_latency(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[["Stdev (ms)", "+/- Stdev (%)"]] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df["Avg (ms)"] -= 1

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = "%" if graph_type is GraphType.PERCENTAGE else "ms"

    ax = df.plot(
        # x="Scheduler",
        use_index=True,
        y="Avg (ms)",
        yerr="Stdev (ms)" if graph_type is not GraphType.PERCENTAGE else None,  # type: ignore[reportArgumentType]
        kind="bar",
        title=f"Scheduler vs. Average Latency{suffix}",
        ylabel=f"Average Latency ({y_units}){suffix}",
        xlabel="Scheduler",
        legend=False,
        rot=45,
    )
    fig = ax.get_figure()
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig.savefig(f"latency-{graph_type.name.lower()}.png", bbox_inches="tight")  # type: ignore[reportUnknownMemberType]


def plot_tail_latency(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[["Stdev (ms)", "+/- Stdev (%)"]] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df["99% (ms)"] -= 1

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = "%" if graph_type is GraphType.PERCENTAGE else "ms"

    ax = df.plot(
        use_index=True,
        y="99% (ms)",
        yerr="Stdev (ms)" if graph_type is not GraphType.PERCENTAGE else None,  # type: ignore[reportArgumentType]
        kind="bar",
        title=f"Scheduler vs. 99% tail latency{suffix}",
        ylabel=f"99% Tail Latency ({y_units}){suffix}",
        xlabel="Scheduler",
        legend=False,
        rot=45,
    )
    fig = ax.get_figure()
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig.savefig(f"tail-latency-{graph_type.name.lower()}.png", bbox_inches="tight")  # type: ignore[reportUnknownMemberType]


def plot_requests(df: pd.DataFrame, graph_type: GraphType) -> None:
    if graph_type is GraphType.RELATIVE:
        baseline: pd.Series[float] = df.loc["eevdf"].copy()  # type: ignore[reportAssignmentType]
        baseline[["Stdev", "+/- Stdev (%)"]] = 0
        df = df - baseline
    elif graph_type is GraphType.PERCENTAGE:
        baseline: pd.Series[float] = df.loc["eevdf"]  # type: ignore[reportAssignmentType]
        df = df.drop("eevdf") / baseline  # type: ignore[reportUnknownMemberType]
        df["Avg"] -= 1

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = " (%)" if graph_type is GraphType.PERCENTAGE else ""

    ax = df.plot(
        use_index=True,
        y="Avg",
        yerr="Stdev" if graph_type is not GraphType.PERCENTAGE else None,  # type: ignore[reportArgumentType]
        kind="bar",
        title=f"Scheduler vs. Average Requests/Second{suffix}",
        ylabel=f"Average Requests/Second{y_units}{suffix}",
        xlabel="Scheduler",
        legend=False,
        rot=45,
    )
    _ = ax.set_xticklabels(ax.get_xticklabels(), ha="right")  # type: ignore[reportUnknownMemberType]
    fig = ax.get_figure()
    fig.savefig(f"requests{graph_type.name.lower()}.png", bbox_inches="tight")  # type: ignore[reportUnknownMemberType]


latency_rows: list[list[str | float]] = []
request_rows: list[list[str | float]] = []
for result in newest.iterdir():
    with result.open() as f:
        contents = f.read()
        lat_m = latency_pat.search(contents)
        req_m = rps_pat.search(contents)
        dist_m = dist_pat.finditer(contents)
        if lat_m is None or req_m is None:
            raise ValueError(f"Invalid results file {result}")

        sched = result.stem
        latency_rows.append(extract_row(sched, lat_m, dist_m))
        request_rows.append(extract_row(sched, lat_m))


lat_col_names = [
    "Scheduler",
    "Avg (ms)",
    "Stdev (ms)",
    "Max (ms)",
    "+/- Stdev (%)",
    "50% (ms)",
    "75% (ms)",
    "90% (ms)",
    "99% (ms)",
]
latency = pd.DataFrame(data=latency_rows, columns=lat_col_names)
latency = latency.set_index("Scheduler")  # type: ignore[reportUnknownMemberType]

req_col_names = ["Scheduler", "Avg", "Stdev", "Max", "+/- Stdev (%)"]
requests = pd.DataFrame(data=request_rows, columns=req_col_names)
requests = requests.set_index("Scheduler")  # type: ignore[reportUnknownMemberType]

print(latency)
print(requests)

plot_latency(latency, GraphType.RELATIVE)
plot_latency(latency, GraphType.PERCENTAGE)
plot_tail_latency(latency, GraphType.RELATIVE)
plot_tail_latency(latency, GraphType.PERCENTAGE)
plot_requests(requests, GraphType.RELATIVE)
plot_requests(requests, GraphType.PERCENTAGE)
