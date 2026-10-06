"""Satiation and storage metrics computed from evaluation traces (true internal state)."""
import numpy as np


def intake_rate(consumed, window=100):
    """Bites per step, summed over variables, in non-overlapping windows."""
    total = consumed.sum(1)
    n = len(total) // window
    return total[: n * window].reshape(n, window).mean(1)


def time_near_setpoint(h, setpoint, tol=2.0):
    """Fraction of time each variable is within tol of its setpoint, averaged over
    variables. (Requiring all variables at once is too strict: commuting between
    resources makes them oscillate out of phase.)"""
    return float((np.abs(h - setpoint) < tol).mean())


def overshoot(h, setpoint):
    """Mean excess above setpoint, averaged over steps and variables."""
    return float(np.clip(h - setpoint, 0, None).mean())


def resumption_latency(h, consumed, setpoint, theta=1.0):
    """After h_i drops below setpoint - theta (downward crossing), steps until the next
    bite of variable i. Returns (median latency, fraction of crossings followed by a
    bite before the end of the trace). NaN latency if there was no crossing."""
    latencies, n_events, n_resumed = [], 0, 0
    thr = setpoint - theta
    for i in range(h.shape[1]):
        below = h[:, i] < thr
        crossings = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        bites = np.flatnonzero(consumed[:, i] > 0)
        for c in crossings:
            n_events += 1
            nxt = bites[bites >= c]
            if len(nxt):
                n_resumed += 1
                latencies.append(nxt[0] - c)
    lat = float(np.median(latencies)) if latencies else np.nan
    return lat, (n_resumed / n_events if n_events else np.nan)


def satiation_summary(trace, setpoint, tol=2.0, theta=1.0):
    h, consumed = trace["h"], trace["consumed"]
    lat, resumed = resumption_latency(h, consumed, setpoint, theta)
    return {
        "near_setpoint": time_near_setpoint(h, setpoint, tol),
        "abs_dev": float(np.abs(h - setpoint).mean()),
        "overshoot": overshoot(h, setpoint),
        "intake": float(consumed.sum(1).mean()),
        "resume_latency": lat,
        "resume_frac": resumed,
        "deaths": int(trace["dead"].sum()),
    }


def storage_summary(trace, setpoint, lookback=5):
    """Storage = mean excess of h above setpoint just before scarcity onset, versus
    the same quantity outside warning/scarcity periods (baseline)."""
    h, sc, warn = trace["h"], trace["scarcity"], trace["warning"]
    onsets = np.flatnonzero(sc[1:] & ~sc[:-1]) + 1
    pre = [h[o - lookback:o].mean() for o in onsets if o >= lookback]
    calm = ~(sc | warn)
    # a scarcity is survived if no death occurs between its onset and its end
    survived = []
    for o in onsets:
        end = o + np.argmax(~sc[o:]) if (~sc[o:]).any() else len(sc)
        survived.append(not trace["dead"][o:end].any())
    return {
        "pre_scarcity_level": float(np.mean(pre)) - setpoint if pre else np.nan,
        "baseline_level": float(h[calm].mean()) - setpoint if calm.any() else np.nan,
        "survival": float(np.mean(survived)) if survived else np.nan,
        "deaths": int(trace["dead"].sum()),
    }
