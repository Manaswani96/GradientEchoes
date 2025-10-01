from gradient_echoes.core.objective import Objective
from gradient_echoes.core.schedules import OneCycle
from gradient_echoes.classical import SGD
from gradient_echoes.core.callbacks import LogEvery

f = lambda x: (x - 4.0) ** 2
g = lambda x: 2.0 * (x - 4.0)
obj = Objective(f, g, init=0.0)

sched = OneCycle(total_steps=200, max_lr=0.4, pct_start=0.3)
opt = SGD(lr=sched, momentum=0.9, nesterov=True)
log = LogEvery(20, prefix="[onecycle]")
x, hist = opt.minimize(obj, steps=200, callback=log)
print("x* =", round(x, 6), "f* =", round(hist[-1]["f"], 8))
