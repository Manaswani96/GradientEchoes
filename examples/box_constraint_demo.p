from gradient_echoes.core.objective import Objective, Constraint
from gradient_echoes.classical import Adam
from gradient_echoes.core.callbacks import LogEvery

# minimize (x-2)^2 subject to x in [0, 1]  -> solution should clamp to 1.0
f = lambda x: (x - 2.0) ** 2
g = lambda x: 2.0 * (x - 2.0)

def project_box(x, lo=0.0, hi=1.0):
    return max(min(float(x), hi), lo)

cnstr = Constraint(name="box[0,1]", project=project_box)
obj = Objective(f, g, init=0.5, constraint=cnstr)

opt = Adam(lr=0.2)
log = LogEvery(10, prefix="[box]")
x, hist = opt.minimize(obj, steps=80, callback=log)
print("x* =", round(x, 6), "f* =", round(hist[-1]['f'], 8))
