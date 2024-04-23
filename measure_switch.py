from pathlib import Path
from timeit import default_timer as timer
import time

from schedman import start_scheduler, stop_scheduler, switch_scheduler

kfile = Path("/sys/kernel/sched_ext/state")


def wait_enabled():
    while kfile.read_text() != "enabled\n":
        time.sleep(0.1)


def wait_disabled():
    while kfile.read_text() != "disabled\n":
        time.sleep(0.1)


wait_disabled()

start = timer()
p_sched = start_scheduler("scx_simple -f")
wait_enabled()

before_switch = timer()
print("First", before_switch - start)

p_sched = switch_scheduler("scx_simple", p_sched)
wait_enabled()  # Should go through right away b/c we can't tell when it turned off then on again

after_switch = timer()
print("Switch", after_switch - before_switch)

_ = stop_scheduler(p_sched)
after = timer()
wait_disabled()

print("Stop", after - after_switch)
