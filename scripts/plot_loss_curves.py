from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import SGD, Adam, RMSProp, AdaGrad
from gradient_echoes.core.callbacks import History
from gradient_echoes.vis.plotting import plot_histories

# quadratic problem
f = lambda x: (x - 3.0)**2
g = lambda x: 2*(x - 3.0)
obj = Objective(f, g, init=0.0)

runs = {}
for name, opt in [
    ("SGD (Nesterov)", SGD(lr=0.15, momentum=0.9, nesterov=True)),
    ("Adam", Adam(lr=0.08)),
    ("RMSProp", RMSProp(lr=0.08)),
    ("AdaGrad", AdaGrad(lr=0.5)),
]:
    H = History()
    _, _ = opt.minimize(obj, steps=150, callback=H)
    runs[name] = H.records

plot_histories(runs, y="f", title="Optimizer loss curves (quadratic)", savepath="docs/assets/loss_curves.png")
print("Saved docs/assets/loss_curves.png")
