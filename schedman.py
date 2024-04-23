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
    return subprocess.Popen(shlex.split(sched_cmd))


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


class CFS:
    timeslice_s: float
    scheds: list[str]

    timer: threading.Timer

    curr_i: int
    curr_sched: str
    curr_p_sched: subprocess.Popen[bytes]

    def __init__(self, scheds: list[str], timeslice_s: float = 0.5) -> None:
        super().__init__()
        # Starting info
        self.timeslice_s = timeslice_s
        self.scheds = scheds
        self.curr_i = 0
        self.curr_sched = scheds[self.curr_i]
        # Start desired sched
        self.curr_p_sched = start_scheduler(self.curr_sched)

        # Start swiching in `timeslice` seconds
        self.timer = self._create_timer()
        self.timer.start()

    def _create_timer(self) -> threading.Timer:
        return threading.Timer(self.timeslice_s, self._update)

    def _update(self) -> None:
        self.curr_i = (self.curr_i + 1) % len(self.scheds)
        self.curr_sched = self.scheds[self.curr_i]
        self.curr_p_sched = switch_scheduler(self.curr_sched, self.curr_p_sched)
        self.timer = self._create_timer()
        self.timer.start()
