import json
import math
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable

import numpy as np

import schedman
from categories import DocGroup
from main import ask
from test_types import TestType

RESULTS = Path("./results")
RESULTS.mkdir(exist_ok=True)

dt = datetime.strftime(datetime.now(), "%Y-%m-%d-%H.%M.%S")
CURR_RESULTS = RESULTS / dt
CURR_RESULTS.mkdir()


def write_output(sched: str, tt: TestType, result: bytes) -> None:
    if sched == "warmup":
        return

    sched_folder = CURR_RESULTS / sched
    sched_folder.mkdir(exist_ok=True)

    ext = "txt" if tt is not TestType.BACKEND else "json"
    _ = (sched_folder / f"{tt.name.lower()}.{ext}").write_bytes(result)


def run_test(tt: TestType, test_func: Callable[[], bytes]) -> None:
    for sched in schedman.SCHEDS:
        print(f"Starting {sched}")
        p_sched = schedman.start_scheduler(sched)

        res = test_func()

        ret = schedman.stop_scheduler(p_sched)

        if ret != 0:
            print(f"Failed to kill {sched}")
            exit(ret)

        print(f"Killed {sched}")
        # Wait for any of last test's connections to finish up
        time.sleep(5)

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
        p99=np.percentile(latency, 99).item(),
    )
    rps = np.array(thread_reqs) / TEST_RUNTIME
    rps_stats = dict(
        avg=np.mean(rps).item(),
        std=np.std(rps).item(),
        max=np.max(rps).item(),
    )
    stats = dict(latency=latency_stats, rps=rps_stats)

    return json.dumps(stats, indent=2).encode()


run_test(TestType.FULL, full)
run_test(TestType.FRONTEND, frontend)
run_test(TestType.BACKEND, backend)
