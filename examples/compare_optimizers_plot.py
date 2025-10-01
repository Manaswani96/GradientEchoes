# examples/compare_optimizers_plot.py
# Compare multiple optimizers on the same quadratic and draw curves.
from gradient_echoes.core.objective import Objective
from gradient_echoes.core.callbacks import History
from gradient_echoes.classical import SGD, Adam, RMSProp, AMSGrad, LBFGS, AdaGrad
from gradient_echoes.vis.plotting import plot_histories

# problem: (x - 3)^2
f = lambda x: (x - 3.0) ** 2
g = lambda x: 2.0 * (x - 3.0)
obj = Objective(f, g, init=0.0)

runs = {}
# each optimizer runs and records its own history
for name, opt in [
    ("SGD(nesterov)", SGD(lr=0.15, momentum=0.9, nesterov=True)),
    ("Adam", Adam(lr=0.08)),
    ("AMSGrad", AMSGrad(lr=0.08)),
    ("RMSProp", RMSProp(lr=0.08, centered=True)),
    ("AdaGrad", AdaGrad(lr=0.5)),
    ("LBFGS", LBFGS(history_size=7)),
]:
    H = History()
    x, hist = opt.minimize(obj, steps=120, callback=H)
    runs[name] = H.records

# plot loss vs steps
plot_histories(runs, y="f", title="Gradient Echoes: loss vs steps", savepath="compare_loss.png")
print("Saved plot to compare_loss.png")

# optional: plot gradient norm^2
plot_histories(runs, y="grad_norm2", title="||grad||^2 vs steps", savepath="compare_gradnorm2.png")
print("Saved plot to compare_gradnorm2.png")
