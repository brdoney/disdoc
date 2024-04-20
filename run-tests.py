from pathlib import Path
import subprocess
import shlex
from datetime import datetime
import time

SCHEDS = [
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

for sched in SCHEDS:
    print(f"Starting {sched} test")

    sched_cmd = f"sudo {sched}"
    p_sched = subprocess.Popen(shlex.split(sched_cmd))

    wrk_cmd = "wrk -t10 -c20 -d30s --script wrk-script.lua http://localhost:8080/ask"
    res = subprocess.check_output(shlex.split(wrk_cmd))

    # We're done testing the current scheduler, so kill it
    p_sched.terminate()
    ret = p_sched.wait()

    if ret != 0:
        print(f"Failed {sched} test")
        exit(ret)

    print(f"Finished {sched} test")

    _ = (CURR_RESULTS / f"{sched}.txt").write_bytes(res)

    # Wait for any of last test's connections to finish up
    time.sleep(5)
