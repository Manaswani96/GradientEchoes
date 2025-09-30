from gradient_echoes.core.objective import Objective
from gradient_echoes.core.callbacks import History, LogEvery
from gradient_echoes.core.schedules import Warmup, Constant
from gradient_echoes.classical import SGD, Adam, LBFGS

def f(x): return (x - 3.0) ** 2
def g(x): return 2.0 * (x - 3.0)

obj = Objective(f, g, init=0.0)
hist = History(); log = LogEvery(25, prefix="[hello]")

print("== SGD with warmup ==>")
sgd = SGD(lr=Warmup(Constant(0.1), warm=20, factor=0.2), momentum=0.9, nesterov=True)
x1, H1 = sgd.minimize(obj, steps=150, callback=lambda t,x,i: (hist(t,x,i), log(t,x,i)))
print("x* =", round(x1, 6), "f* =", round(H1[-1]["f"], 8))

print("\n== AdamW ==>")
adam = Adam(lr=0.05, weight_decay=0.01)
x2, H2 = adam.minimize(obj, steps=120, callback=lambda t,x,i: (hist(t,x,i), log(t,x,i)))
print("x* =", round(x2, 6), "f* =", round(H2[-1]["f"], 8))

print("\n== L-BFGS (toy) ==>")
lb = LBFGS(history_size=7)
x3, H3 = lb.minimize(obj, steps=50, callback=lambda t,x,i: (hist(t,x,i), log(t,x,i)))
print("x* =", round(x3, 6), "f* =", round(H3[-1]["f"], 8))
