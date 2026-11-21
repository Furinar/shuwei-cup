"""Static angular calibration for Question 3. Run with ../.venv-mathmodel/bin/python.

The reference direction is fixed along +x; no positions or paths are estimated.
OpenAI Codex assisted with implementation and numerical checks, 2026-11-20.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, roots_legendre


OUT = Path(__file__).resolve().parent
R = 40.0
OFFSET, CONTROL, DELTA = np.deg2rad([0.8, 0.6, 5.0])
TAU = np.hypot(OFFSET, CONTROL)
REQUIRED = 0.95
SAMPLES = 2_000_000
SEED = 20261120
TRUNCATION = 10.0


def normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


class SectorIntegral:
    """Independent standard normals for instrument angle and rotated x noise."""

    def __init__(self, order):
        nodes, weights = roots_legendre(order)
        self.nodes, self.weights = nodes, weights
        z = TRUNCATION * nodes
        self.z_weights = TRUNCATION * weights * normal_pdf(z)
        self.cosine = np.cos(TAU * z)[:, None]
        self.sine = np.sin(TAU * z)[:, None]

    def hit_probability(self, sigma):
        # For a fixed instrument angle e, the rotated coordinates, divided by
        # sigma, are independent normals with means rho*cos(e), rho*sin(e).
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive, finite coordinate standard deviation required")
        rho = R / sigma
        mx, my = rho * self.cosine, rho * self.sine
        # Inside the forward sector iff x >= 0 and -x*tan(delta) <= y <= x*tan(delta).
        lower_x = np.clip(-mx, -TRUNCATION, TRUNCATION)
        half = (TRUNCATION - lower_x) / 2.0
        u = lower_x + half * (self.nodes + 1.0)
        wx = half * self.weights * normal_pdf(u)
        width = (mx + u) * np.tan(DELTA)
        upper, lower = width - my, -width - my
        # Use upper tails for positive limits to avoid subtracting two numbers near 1.
        cond = np.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))
        return float(np.sum(self.z_weights * np.sum(wx * cond, axis=1)))

    def threshold(self):
        return brentq(lambda sig: self.hit_probability(sig) - REQUIRED, 1.0, 2.0,
                      xtol=1e-13, rtol=1e-14)


def coordinate_sd(s):
    return 0.3 + 0.02 * s


def first_order(sigma):
    total = np.hypot(sigma / R, TAU)
    return float(1.0 - 2.0 * ndtr(-DELTA / total))


def wilson(hits, n):
    z = ndtri(0.975)
    rate = hits / n
    scale = 1 + z**2 / n
    center = (rate + z**2 / (2*n)) / scale
    radius = z * np.sqrt(rate*(1-rate)/n + z**2/(4*n**2)) / scale
    return [float(max(0.0, center-radius)), float(min(1.0, center+radius))]


def monte_carlo(scales):
    """Direct simulation from all four independent stated Gaussian measurements."""
    rng = np.random.default_rng(SEED)
    hits = np.zeros(len(scales), dtype=int)
    for start in range(0, SAMPLES, 100_000):
        size = min(100_000, SAMPLES - start)
        x, y, z_offset, z_control = rng.standard_normal((4, size))
        instrument = OFFSET * z_offset + CONTROL * z_control
        for i, sigma in enumerate(scales):
            observed = np.arctan2(sigma*y, R + sigma*x) + instrument
            wrapped = np.remainder(observed + np.pi, 2*np.pi) - np.pi
            hits[i] += np.count_nonzero(np.abs(wrapped) <= DELTA)
    return [
        {"hits": int(k), "n": SAMPLES, "rate": k/SAMPLES, "wilson_95": wilson(k, SAMPLES)}
        for k in hits
    ]


def make_plot(integral, s_limit, checks, path):
    # Numerical x values are independent calibration settings; no field data.
    x = np.linspace(0.0, 85.0, 171)
    prob = np.array([integral.hit_probability(coordinate_sd(s)) for s in x])
    approx = np.array([first_order(coordinate_sd(s)) for s in x])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    left, right = axes
    left.plot(x, prob, color="#155970", label="Exact sector integral")
    left.axhline(REQUIRED, color="#727272", lw=1, ls="--")
    left.axvline(s_limit, color="#727272", lw=1, ls=":")
    left.annotate(f"{s_limit:.2f} m", (s_limit, 0.95), xytext=(-38, -23),
                  textcoords="offset points", fontsize=9)
    mc = [p["monte_carlo"] for p in checks]
    rates = np.array([p["rate"] for p in mc])
    intervals = np.array([p["wilson_95"] for p in mc]).T
    left.errorbar([p["s_m"] for p in checks], rates,
                  yerr=np.vstack((rates - intervals[0], intervals[1] - rates)),
                  fmt="o", ms=4, capsize=3, color="#a84621", label="Direct simulation (95% CI)")
    left.set(xlabel="Calibration setting s (m)", ylabel="Probability within ±5°",
             title="A. Angular coverage", ylim=(0.895, 1.006))
    left.legend(fontsize=8, loc="lower left", frameon=False)
    right.plot(x, (approx-prob)*100, color="#754c86")
    right.set(xlabel="Calibration setting s (m)", ylabel="Approximation minus exact (percentage points)",
              title="B. First-order approximation error")
    for ax in axes:
        ax.grid(axis="y", alpha=0.20)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def main():
    integral = SectorIntegral(96)
    limit = integral.threshold()
    s_limit = (limit - 0.3) / 0.02
    approx_limit = float(R * np.sqrt((DELTA / ndtri((1+REQUIRED)/2))**2 - TAU**2))
    approx_s = (approx_limit - 0.3) / 0.02
    settings = [0.0, 40.0, s_limit, 80.0]
    scales = [coordinate_sd(s) for s in settings]
    simulations = monte_carlo(scales)
    checks = []
    for s, sigma, mc in zip(settings, scales, simulations):
        variance = np.array([(sigma/R)**2, OFFSET**2, CONTROL**2])
        checks.append({
            "s_m": s, "sigma_m": sigma, "p_exact": integral.hit_probability(sigma),
            "p_first_order": first_order(sigma), "monte_carlo": mc,
            "first_order_variance_fractions": dict(zip(
                ["position", "zero_offset", "control"], (variance / variance.sum()).tolist())),
        })
    perturbation = 0.1
    step = 1e-4
    slope = (integral.hit_probability(limit+step) - integral.hit_probability(limit-step)) / (2*step)
    orders = {str(n): SectorIntegral(n).threshold() for n in (64, 96, 128)}
    below = integral.hit_probability(coordinate_sd(s_limit-perturbation))
    above = integral.hit_probability(coordinate_sd(s_limit+perturbation))
    assert below > REQUIRED > above
    assert abs(integral.hit_probability(limit)-REQUIRED) < 1e-12
    assert max(abs(est-limit) for est in orders.values()) < 5e-10
    result = {
        "nominal_range_m": R, "offset_sd_deg": float(np.rad2deg(OFFSET)),
        "control_sd_deg": float(np.rad2deg(CONTROL)), "tolerance_deg": float(np.rad2deg(DELTA)),
        "required_probability": REQUIRED, "sigma_limit_m": limit, "s_limit_m": s_limit,
        "admissible_at_80m": {"a_min_m": 0, "b_min": 0, "a_plus_80b_max_m": limit},
        "first_order_sigma_limit_m": approx_limit, "first_order_s_limit_m": approx_s,
        "first_order_distance_overestimate_m": approx_s-s_limit,
        "p_slope_per_m_sigma_at_threshold": slope, "p_slope_per_m_s_at_threshold": 0.02*slope,
        "quadrature_sigma_limit_by_order": orders,
        "gaussian_tail_truncation_probability_bound": float(4*ndtr(-TRUNCATION)),
        "threshold_bracket": {"offset_s_m": perturbation, "p_below": below, "p_above": above},
        "mc_seed": SEED, "mc_trials_per_setting": SAMPLES, "common_random_numbers_across_settings": True,
        "checks": checks,
    }
    make_plot(integral, s_limit, checks, OUT/"q3_validation.pdf")
    (OUT/"q3_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"sigma_limit={limit:.10f} m, s_limit={s_limit:.8f} m")
    for item in checks:
        print(f"s={item['s_m']:.6f} p_exact={item['p_exact']:.10f} MC={item['monte_carlo']}")


if __name__ == "__main__":
    main()
