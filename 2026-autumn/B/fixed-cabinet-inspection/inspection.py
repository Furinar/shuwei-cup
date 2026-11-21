"""Compute the fixed-plate imaging region and isothermal instrument estimates.

Inputs and outputs are siblings of this file by default. The two cross-checks
use projected image-edge coordinates and a full covariance matrix respectively.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plate_region(settings, min_pixels):
    d_min = settings["distance_min_m"]
    d_max = settings["distance_max_m"]
    offset_limit = settings["offset_limit_m"]
    half_plate = settings["plate_width_m"] / 2
    tangent = math.tan(math.radians(settings["horizontal_fov_deg"]) / 2)
    pixel_scale = settings["sensor_pixels"] * half_plate / tangent
    left = max(d_min, half_plate / tangent)
    right = min(d_max, pixel_scale / min_pixels)
    mechanical_area = (d_max - d_min) * 2 * offset_limit

    def primitive(d):
        clearance = tangent * d - half_plate
        return (max(0, clearance)**2 - max(0, clearance - offset_limit)**2) / tangent

    if right < left:
        return {"min_pixels": min_pixels, "area_m2": 0.0, "fraction": 0.0, "best": None}
    area = primitive(right) - primitive(left)
    clearance = max(0.0, tangent * left - half_plate)
    return {
        "min_pixels": min_pixels,
        "distance_bounds_m": [left, right],
        "offset_saturation_distance_m": (offset_limit + half_plate) / tangent,
        "area_m2": area,
        "fraction": area / mechanical_area,
        "best": {
            "distance_m": left,
            "offset_m": 0.0,
            "plate_pixels": pixel_scale / left,
            "admissible_offset_limit_m": min(offset_limit, clearance),
            "image_clearance_per_side_m": clearance,
            "image_clearance_per_side_pixels": (settings["sensor_pixels"] - pixel_scale / left) / 2,
        },
    }


def grid_areas(settings, thresholds):
    """Pixel-space midpoint raster using the two projected plate edge positions."""
    n_distance = settings["grid_distance_cells"]
    n_offset = settings["grid_offset_cells"]
    d_min, d_max = settings["distance_min_m"], settings["distance_max_m"]
    limit = settings["offset_limit_m"]
    pixels = settings["sensor_pixels"]
    focal_px = pixels / (2 * math.tan(math.radians(settings["horizontal_fov_deg"]) / 2))
    d_step = (d_max - d_min) / n_distance
    u_step = 2 * limit / n_offset
    distances = d_min + (np.arange(n_distance) + 0.5) * d_step
    offsets = -limit + (np.arange(n_offset) + 0.5) * u_step
    half_plate = settings["plate_width_m"] / 2
    counts = {threshold: 0 for threshold in thresholds}
    for start in range(0, len(distances), 32):
        d = distances[start:start + 32, None]
        left_px = pixels / 2 + focal_px * (-half_plate - offsets) / d
        right_px = pixels / 2 + focal_px * (half_plate - offsets) / d
        inside = (left_px >= 0) & (right_px <= pixels)
        width = right_px - left_px
        for threshold in thresholds:
            counts[threshold] += int(np.count_nonzero(inside & (width >= threshold)))
    return {threshold: count * d_step * u_step for threshold, count in counts.items()}


def analyze_plate(settings):
    thresholds = sorted(set([settings["min_plate_pixels"], *settings["sensitivity_min_pixels"]]))
    grid = grid_areas(settings, thresholds)
    rows = []
    for threshold in thresholds:
        row = plate_region(settings, threshold)
        row["grid_area_m2"] = grid[threshold]
        row["grid_abs_difference_m2"] = abs(row["area_m2"] - grid[threshold])
        rows.append(row)
    return {
        "mechanical_area_m2": 2 * settings["offset_limit_m"] * (
            settings["distance_max_m"] - settings["distance_min_m"]),
        "grid_cells": [settings["grid_distance_cells"], settings["grid_offset_cells"]],
        "nominal_min_pixels": settings["min_plate_pixels"],
        "by_threshold": rows,
    }


def estimate_pair(pair, settings):
    observations = np.array([channel["reading_C"] for channel in pair])
    variances = np.array([channel["repeat_sd_C"]**2 for channel in pair])
    common_var = settings["common_reference_sd_C"]**2
    precisions = 1 / variances
    weights = precisions / precisions.sum()
    total_var = common_var + 1 / precisions.sum()
    mean = float(weights @ observations)
    z = NormalDist().inv_cdf((1 + settings["confidence"]) / 2)
    radius = z * math.sqrt(total_var)
    # Independently form GLS from the joint marginal covariance of the readings.
    covariance = np.diag(variances) + common_var * np.ones((2, 2))
    gls = np.linalg.solve(covariance, np.ones(2))
    gls /= gls.sum()
    np.testing.assert_allclose(weights, gls, rtol=1e-12)
    np.testing.assert_allclose(total_var, gls @ covariance @ gls, rtol=1e-12)
    cost = sum(channel["setup_min"] for channel in pair)
    return {
        "pair": [channel["id"] for channel in pair],
        "cost_min": cost,
        "feasible_at_nominal_budget": cost <= settings["budget_instrument_min"],
        "weights": weights.tolist(),
        "mean_C": mean,
        "variance_C2": float(total_var),
        "standard_error_C": math.sqrt(total_var),
        "interval_C": [mean - radius, mean + radius],
    }


def analyze_temperature(settings):
    rows = [estimate_pair(pair, settings) for pair in itertools.combinations(settings["channels"], 2)]
    best = {}
    for budget in sorted(set([settings["budget_instrument_min"], *settings["sensitivity_budgets"]])):
        feasible = [row for row in rows if row["cost_min"] <= budget]
        if not feasible:
            best[str(budget)] = None
            continue
        best[str(budget)] = min(feasible, key=lambda row: (row["variance_C2"], row["cost_min"], row["pair"]))
    return {
        "all_pairs": rows,
        "confidence": settings["confidence"],
        "budget_instrument_min": settings["budget_instrument_min"],
        "best_by_budget": best,
        "gls_check": "All two-channel weights and variances matched matrix GLS at relative tolerance 1e-12.",
    }


def plot_region(settings, result, path):
    row = next(r for r in result["by_threshold"] if r["min_pixels"] == result["nominal_min_pixels"])
    d_min, d_max = settings["distance_min_m"], settings["distance_max_m"]
    offset = settings["offset_limit_m"]
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    ax.set(xlim=(d_min - .02 * (d_max - d_min), d_max), ylim=(-1.07 * offset, 1.07 * offset),
           xlabel="Normal distance from the plate, d (m)", ylabel="Lateral mounting offset, u (m)")
    ax.set_facecolor("#f6f7f8")
    ax.axvline(d_min, color="#5d6775", lw=.8)
    if row["best"] is not None:
        left, right = row["distance_bounds_m"]
        tangent = math.tan(math.radians(settings["horizontal_fov_deg"]) / 2)
        switch = row["offset_saturation_distance_m"]
        x = np.sort(np.clip([left, switch, right], left, right))
        y = np.clip(x * tangent - settings["plate_width_m"] / 2, 0, offset)
        ax.fill_between(x, -y, y, facecolor="#b4dcd5", edgecolor="#167565", lw=1.5,
                        label=f"Feasible at {row['min_pixels']} pixels")
        ax.axvline(right, color="#485c76", ls="--", lw=1.1)
        ax.plot(left, 0, marker="o", color="#9d3e2f", markersize=6, label="Centered optimum", zorder=3)
    ax.legend(loc="upper right", frameon=True, framealpha=.96)
    ax.tick_params(direction="out")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=folder / "inspection_inputs.json")
    parser.add_argument("--outdir", type=Path, default=folder)
    args = parser.parse_args()
    data = json.loads(args.input.read_bytes())
    plate = analyze_plate(data["plate"])
    temperature = analyze_temperature(data["temperature"])
    results = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "plate": plate, "temperature": temperature,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "inspection_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_region(data["plate"], plate, args.outdir / "label_region.pdf")


if __name__ == "__main__":
    main()
