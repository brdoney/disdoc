import re
from pathlib import Path
from typing import no_type_check
import pandas as pd
import matplotlib.pyplot as plt

from parse_utils import GraphType, remove_suffix

plt.style.use("seaborn-v0_8")

TLATENCY_PAT = re.compile(r"(\d+%)\s+(\S+m?s)")
RPS_PAT = re.compile(r"Requests\/sec:\s+(\S+)")

RESULTS_DIR = Path("./better-cfs-results/")

cols: dict[str, list[str | float]] = {
    "Scheduler": [],
    "99% (ms)": [],
    "RPS": [],
    "Timeslice": [],
}


def parse_wrk(result_file: Path) -> None:
    with result_file.open() as f:
        scheduler = result_file.parent.stem
        cols["Scheduler"].append(scheduler)

        timeslice = remove_suffix(result_file.stem)
        cols["Timeslice"].append(timeslice)

        contents = f.read()
        dist_m = TLATENCY_PAT.finditer(contents)
        reqt_m = RPS_PAT.search(contents)

        if reqt_m is None:
            raise ValueError(f"Invalid results file {result_file}")

        for m in dist_m:
            if m.group(1) == "99%":
                cols["99% (ms)"].append(remove_suffix(m.group(2)))

        cols["RPS"].append(float(reqt_m.group(1)))


def parse_eevdf(eevdf_file: Path) -> tuple[float, float]:
    with eevdf_file.open() as f:
        contents = f.read()
        dist_m = TLATENCY_PAT.finditer(contents)
        reqt_m = RPS_PAT.search(contents)

        if reqt_m is None:
            raise ValueError(f"Invalid results file {result_file}")

        tail_latency = 0
        for m in dist_m:
            if m.group(1) == "99%":
                tail_latency = remove_suffix(m.group(2))

        rps = float(reqt_m.group(1))

        return tail_latency, rps


def plot_rps(df: pd.DataFrame, graph_type: GraphType, eevdf_rps: float) -> None:
    df = df.copy()
    if graph_type is GraphType.RELATIVE:
        df["RPS"] = df["RPS"] - eevdf_rps
    elif graph_type is GraphType.PERCENTAGE:
        df["RPS"] = (df["RPS"] / eevdf_rps - 1) * 100

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = " (%)" if graph_type is GraphType.PERCENTAGE else ""

    fig, ax = plt.subplots()

    if graph_type is not GraphType.ABSOLUTE:
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=1)

    df = df.set_index("Timeslice").sort_index().groupby("Scheduler")
    _ = df["RPS"].plot(
        title=f"Scheduler vs. Requests/Second{suffix}",
        ylabel=f"Requests/Second{y_units}{suffix} (higher is better)",
        xlabel="Timeslice",
        legend=True,
        ax=ax,
    )

    fig.savefig(  # type: ignore[reportUnknownMemberType]
        f"complete-rps-{graph_type.name.lower()}.png",
        bbox_inches="tight",
        dpi=200,
    )
    fig.clf()


def plot_tail_latency(
    df: pd.DataFrame, graph_type: GraphType, eevdf_tail_latency: float
) -> None:
    df = df.copy()
    if graph_type is GraphType.RELATIVE:
        df["99% (ms)"] = df["99% (ms)"] - eevdf_tail_latency
    elif graph_type is GraphType.PERCENTAGE:
        df["99% (ms)"] = (df["99% (ms)"] / eevdf_tail_latency - 1) * 100

    suffix: str = " Relative to EEVDF" if graph_type is not GraphType.ABSOLUTE else ""
    y_units: str = "%" if graph_type is GraphType.PERCENTAGE else "ms"

    fig, ax = plt.subplots()

    if graph_type is not GraphType.ABSOLUTE:
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=1)

    df = df.set_index("Timeslice").sort_index().groupby("Scheduler")
    _ = df["99% (ms)"].plot(
        title=f"Scheduler vs. 99% Tail Latency{suffix}",
        ylabel=f"99% Tail Latency ({y_units}){suffix} (lower is better)",
        xlabel="Timeslice",
        legend=True,
        ax=ax,
    )
    fig.savefig(  # type: ignore[reportUnknownMemberType]
        f"complete-tail-latency-{graph_type.name.lower()}.png",
        bbox_inches="tight",
        dpi=200,
    )
    fig.clf()


for sched_dir in RESULTS_DIR.iterdir():
    if not sched_dir.is_dir():
        # Don't read EEVDF
        continue

    for result_file in sched_dir.iterdir():
        parse_wrk(result_file)

eevdf_info = parse_eevdf(RESULTS_DIR / "eevdf.txt")

df = pd.DataFrame(cols)
print(df)
print(eevdf_info)


@no_type_check
def make_col_sorted_percent(
    series: pd.Series, eevdf_val: float, invert: bool
) -> pd.Series:
    series = ((series / eevdf_val) - 1) * 100
    if invert:
        series *= -1
    return series.sort_values(ascending=False)


print(make_col_sorted_percent(df["99% (ms)"], eevdf_info[0], True))  # type: ignore[reportUnknownMemberType]
print(make_col_sorted_percent(df["RPS"], eevdf_info[1], False))  # type: ignore[reportUnknownMemberType]

plot_tail_latency(df, GraphType.PERCENTAGE, eevdf_info[0])
plot_rps(df, GraphType.PERCENTAGE, eevdf_info[1])
