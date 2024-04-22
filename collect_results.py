import json
import math
import shlex
import subprocess
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import threading
from timeit import default_timer as timer
from typing import Callable

import numpy as np

from categories import DocGroup
from main import ask

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

RESULTS = Path("./results")
RESULTS.mkdir(exist_ok=True)

dt = datetime.strftime(datetime.now(), "%Y-%m-%d-%H.%M.%S")
CURR_RESULTS = RESULTS / dt
CURR_RESULTS.mkdir()


def start_scheduler(sched: str) -> subprocess.Popen[bytes]:
    print(f"Starting {sched} test")
    if sched == "eevdf" or sched == "warmup":
        sched_cmd = f"echo {sched}"
    else:
        sched_cmd = f"sudo {sched}"
    return subprocess.Popen(shlex.split(sched_cmd))


def stop_scheduler(sched: str, p_sched: subprocess.Popen[bytes]) -> None:
    # We're done testing the current scheduler, so kill it
    p_sched.terminate()
    ret = p_sched.wait()

    if ret != 0:
        print(f"Failed {sched} test")
        exit(ret)

    print(f"Finished {sched} test")

    # Wait for any of last test's connections to finish up
    time.sleep(5)


class TestType(Enum):
    FULL = auto()
    BACKEND = auto()
    FRONTEND = auto()


def write_output(sched: str, tt: TestType, result: bytes) -> None:
    if sched == "warmup":
        return

    sched_folder = CURR_RESULTS / sched
    sched_folder.mkdir(exist_ok=True)

    ext = "txt" if tt is not TestType.BACKEND else "json"
    _ = (sched_folder / f"{tt.name.lower()}.{ext}").write_bytes(result)


def run_test(tt: TestType, test_func: Callable[[], bytes]) -> None:
    for sched in SCHEDS:
        p_sched = start_scheduler(sched)
        res = test_func()
        stop_scheduler(sched, p_sched)
        write_output(sched, tt, res)


def full() -> bytes:
    wrk_cmd = "wrk -t10 -c20 -d30s --latency --script wrk-script.lua http://localhost:8080/ask"
    return subprocess.check_output(shlex.split(wrk_cmd))


def frontend() -> bytes:
    wrk_cmd = "wrk -t10 -c20 -d30s --latency --script wrk-script.lua http://localhost:8080/askecho"
    return subprocess.check_output(shlex.split(wrk_cmd))


def percentile(values: list[float], p: float):
    if not isinstance(p, float) or not (0.0 <= p <= 1.0):
        raise ValueError("p must be a float in the range [0.0; 1.0]")

    values = sorted(values)
    if not values:
        raise ValueError("no value")

    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f != c:
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1
    else:
        return values[int(k)]


def backend() -> bytes:
    # {"question":"internal vs external submissions", "category":"all"}'
    category = DocGroup.from_str("all")
    question = "internal vs external submissions"

    NUM_THREADS = 4
    TEST_RUNTIME = 30

    thread_times: list[list[float]] = [[] for _ in range(NUM_THREADS)]
    thread_reqs: list[float] = [0] * NUM_THREADS
    shutdown_ev = threading.Event()

    def time_ask(id: int) -> None:
        while not shutdown_ev.is_set():
            before = timer()
            _ = ask(question, category)
            after = timer()
            thread_reqs[id] += 1
            # Convert to ms
            delta = (after - before) * 1000
            thread_times[id].append(delta)

    handles: list[threading.Thread] = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=time_ask, args=(i,))
        t.start()
        handles.append(t)

    time.sleep(TEST_RUNTIME)
    shutdown_ev.set()

    for t in handles:
        t.join()

    latency = np.hstack(thread_times)
    latency_stats = dict(
        avg=np.mean(latency).item(),
        max=np.max(latency).item(),  # type: ignore[reportAny]
        std=np.std(latency).item(),
        p50=np.percentile(latency, 50).item(),
        p75=np.percentile(latency, 75).item(),
        p90=np.percentile(latency, 90).item(),
        p95=np.percentile(latency, 95).item(),
    )
    rps = np.array(thread_reqs)
    rps_stats = dict(
        avg=np.mean(rps).item(),
        std=np.std(rps).item(),
        max=np.max(rps).item(),  # type: ignore[reportAny]
    )
    stats = dict(latency=latency_stats, rps=rps_stats)

    return json.dumps(stats, indent=2).encode()


run_test(TestType.FULL, full)
run_test(TestType.FRONTEND, frontend)
run_test(TestType.BACKEND, backend)
