import subprocess
import shlex
import threading

SCHEDS = [
    "warmup",
    # Default (EEVDF)
    "eevdf",
    # One sched
    # "scx_flatcg",  # Seems to stall sometimes?
    "scx_central",
    "scx_nest",
    "scx_pair",
    "scx_qmap",
    "scx_simple",
    "scx_simple -f",
    # User-space scheds
    # "scx_userland",  # Not production ready
    "scx_rustland",
    "scx_rusty",
    # "scx_layered",  # Requires layer spec?
]


def start_scheduler(sched: str) -> subprocess.Popen[bytes]:
    if sched == "eevdf" or sched == "warmup":
        sched_cmd = f"echo {sched}"
    else:
        sched_cmd = f"sudo {sched}"
    return subprocess.Popen(
        shlex.split(sched_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def stop_scheduler(p_sched: subprocess.Popen[bytes]) -> int:
    # We're done testing the current scheduler, so kill it
    p_sched.terminate()
    return p_sched.wait()


def switch_scheduler(
    new_sched: str, p_sched: subprocess.Popen[bytes]
) -> subprocess.Popen[bytes]:
    res = stop_scheduler(p_sched)
    # -15 means sigterm, which we sent to stop it - normally gets intercepted
    # by prog and turned to 0 during unmount, but doesn't happen until it's properly started
    assert res == 0, f"Unexpected signal when stopping scheduler: {res}"
    return start_scheduler(new_sched)


class WeightedCFS:
    """General purpose class for weighted CFS"""

    timeslice_s: float
    scheds: list[str]
    weights: list[float]

    timer: threading.Timer

    curr_i: int
    curr_sched: str
    curr_p_sched: subprocess.Popen[bytes]

    def __init__(
        self, scheds: list[str], weights: list[float], timeslice_s: float
    ) -> None:
        super().__init__()
        # Starting info
        self.timeslice_s = timeslice_s
        self.scheds = scheds
        self.curr_i = 0
        self.curr_sched = scheds[self.curr_i]
        self.weights = weights
        print("weights", [timeslice_s * w for w in weights])
        # Start desired sched
        self.curr_p_sched = start_scheduler(self.curr_sched)

        # Start swiching in `timeslice` seconds
        self.timer = self._create_timer()
        self.timer.start()

    def _create_timer(self) -> threading.Timer:
        return threading.Timer(
            self.timeslice_s * self.weights[self.curr_i], self._update
        )

    def _update(self) -> None:
        self.curr_i = (self.curr_i + 1) % len(self.scheds)
        self.curr_sched = self.scheds[self.curr_i]
        print(f"Switching to {self.curr_sched}")
        self.curr_p_sched = switch_scheduler(self.curr_sched, self.curr_p_sched)
        self.timer = self._create_timer()
        self.timer.start()


class CFS(WeightedCFS):
    def __init__(self, scheds: list[str], timeslice_s: float = 0.75) -> None:
        super().__init__(scheds, [1.0] * len(scheds), timeslice_s)


class DeltaWeightedCFS(WeightedCFS):
    """Prioritises the system with the lowest throughput"""

    def __init__(
        self,
        scheds: list[str],
        frontend_s: float,
        backend_s: float,
        timeslice_s: float = 2,
    ) -> None:
        denom = frontend_s + backend_s
        # First we find how much each system contributes to overall throughput,
        # then invert so most time-constrained gets highest share
        weights = [1 - frontend_s / denom, 1 - backend_s / denom]
        print(weights)
        # These are shares of the comined timeslice (e.g. 75% of the combined timeslices)
        combined_timeslice = 2 * timeslice_s
        super().__init__(scheds, weights, combined_timeslice)


class BenefitWeightedCFS(WeightedCFS):
    """Prioritises the system with the highest benefit"""

    def __init__(
        self,
        scheds: list[str],
        frontend_benefit: float,
        backend_benefit: float,
        timeslice_s: float = 0.75,
    ) -> None:
        denom = frontend_benefit + backend_benefit
        weights = [frontend_benefit / denom, backend_benefit / denom]
        print(weights)
        # These are shares of the combined timeslice
        combined_timeslice = 2 * timeslice_s
        super().__init__(scheds, weights, combined_timeslice)


class DeltaBenefitWeightedCFS(WeightedCFS):
    """Prioritises the system with the highest benefit and lowest throughput"""

    def __init__(
        self,
        scheds: list[str],
        frontend_s: float,
        frontend_benefit: float,
        backend_s: float,
        backend_benefit: float,
        timeslice_s: float = 0.75,
    ) -> None:
        # See DeltaWeighted for explanation
        delta_denom = frontend_s + backend_s
        delta_weights = [1 - frontend_s / delta_denom, 1 - backend_s / delta_denom]

        # See BenefitWeighted for explanation
        benefit_denom = frontend_benefit + backend_benefit
        benefit_weights = [
            frontend_benefit / benefit_denom,
            backend_benefit / benefit_denom,
        ]

        weights = [
            delta_weights[0] * benefit_weights[0],
            delta_weights[1] * benefit_weights[1],
        ]

        # Standardize it (e.g. 0.6*0.2 + 0.4*0.8 != 1.0 anymore)
        total = sum(weights)
        weights = [
            weights[0] / total,
            weights[1] / total,
        ]
        print(weights)

        # These are shares of the combined timeslice
        combined_timeslice = 2 * timeslice_s
        super().__init__(scheds, weights, combined_timeslice)
