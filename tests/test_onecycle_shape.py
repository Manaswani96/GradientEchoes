from gradient_echoes.core.schedules import OneCycle

def test_onecycle_start_peak_end():
    oc = OneCycle(total_steps=10, max_lr=1.0, pct_start=0.4, div_factor=10, final_div_factor=100)
    vals = [oc(t) for t in range(10)]
    assert abs(vals[0] - 0.1) < 1e-6           # start ~ max_lr/div
    assert max(vals) <= 1.0 + 1e-8             # peak <= max_lr
    assert abs(vals[-1] - 0.01) < 1e-6         # end ~ max_lr/final_div
