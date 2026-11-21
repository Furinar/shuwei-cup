"""Worst-case nameplate image coverage for a fixed cabinet camera fixture.

The tolerance levels below are constructed stress cases, not measurements.
Run with the project Python environment or use --input and --output paths.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EQUAL_HALF_WIDTHS_M = (0.0, 0.01, 0.02)


def analyze(s, axial, lateral):
    dmin, dmax = s["distance_min_m"], s["distance_max_m"]
    rail = s["offset_limit_m"]
    half = s["plate_width_m"] / 2
    pixels = s["sensor_pixels"]
    threshold = s["min_plate_pixels"]
    tan_half = math.tan(math.radians(s["horizontal_fov_deg"]) / 2)
    focal = pixels / (2 * tan_half)
    pixel_limit = focal * (2 * half) / threshold
    lower = axial + max(dmin, (half + lateral) / tan_half)
    upper = min(dmax, pixel_limit) - axial
    row = {
        "axial_half_width_m": axial, "lateral_half_width_m": lateral,
        "d_bounds_m": None, "area_m2": 0.0,
        "mechanical_area_fraction": 0.0, "maximin": None,
    }

    if upper < lower or rail < lateral:
        return row

    def u_limit(d):
        return max(0.0, min(rail - lateral, (d - axial) * tan_half - half - lateral))

    # The inner clipping switch of this piecewise linear width has no
    # dependence on the lateral tolerance: it erodes both upper bounds.
    switch = axial + (rail + half) / tan_half
    knots = [lower, *([switch] if lower < switch < upper else []), upper]
    area = sum((b - a) * (u_limit(a) + u_limit(b))
               for a, b in zip(knots, knots[1:]))
    # The far axial face gives the narrowest image. At the nearest face,
    # the most extreme lateral position gives the least image-edge clearance.
    gap_m = (lower - axial) * tan_half - half - lateral
    row.update(
        d_bounds_m=[lower, upper],
        area_m2=area,
        mechanical_area_fraction=area / ((dmax - dmin) * 2 * rail),
        maximin={
            "d_command_m": lower, "u_command_m": 0.0,
            "tied_absolute_offset_limit_m": u_limit(lower),
            "worst_width_px": focal * (2 * half) / (lower + axial),
            "worst_edge_clearance_m": gap_m,
            "worst_edge_clearance_px": focal * gap_m / (lower - axial),
        },
    )
    return row


def check_grid(s, row, shape):
    """Test the projected physical endpoints at all four error-box corners."""
    ed, eu = row["axial_half_width_m"], row["lateral_half_width_m"]
    dmin, dmax, rail = s["distance_min_m"], s["distance_max_m"], s["offset_limit_m"]
    pixels, half = s["sensor_pixels"], s["plate_width_m"] / 2
    tangent = math.tan(math.radians(s["horizontal_fov_deg"]) / 2)
    focal = pixels / (2 * tangent)
    nd, nu = shape
    dd, du = (dmax - dmin) / nd, 2 * rail / nu
    d_values = dmin + dd * (np.arange(nd) + 0.5)
    u_values = -rail + du * (np.arange(nu)[None, :] + 0.5)
    accepted, mismatches = 0, 0
    for start in range(0, nd, 32):
        ds = d_values[start:start + 32, None]
        mask = np.ones((len(ds), nu), dtype=bool)
        for d_shift in sorted({-ed, ed}):
            d = ds + d_shift
            for u_shift in sorted({-eu, eu}):
                u = u_values + u_shift
                left_edge = pixels / 2 + focal * (-half - u) / d
                right_edge = pixels / 2 + focal * (half - u) / d
                mask &= ((d >= dmin) & (d <= dmax) & (np.abs(u) <= rail)
                         & (left_edge >= 0) & (right_edge <= pixels)
                         & (right_edge - left_edge >= s["min_plate_pixels"]))
        accepted += int(mask.sum())
        closed_form = np.zeros_like(mask)
        if row["d_bounds_m"] is not None:
            lo, hi = row["d_bounds_m"]
            radius = np.minimum(rail - eu, (ds - ed) * tangent - half - eu)
            closed_form = (ds >= lo) & (ds <= hi) & (np.abs(u_values) <= radius)
        mismatches += int(np.count_nonzero(mask != closed_form))
    if mismatches:
        raise AssertionError(f"{mismatches} midpoint classifications differ")
    area = accepted * dd * du
    # Vertical integration rounds the width by at most du. Two columns can
    # straddle the distance limits. Interior width is Lipschitz with slope 2t.
    error_bound = (dmax - dmin) * (du + tangent * dd) + 4 * rail * dd
    error = abs(row["area_m2"] - area)
    if error > error_bound:
        raise AssertionError("raster area outside its boundary-cell error bound")
    return {"counts": [nd, nu], "corner_grid_area_m2": area,
            "absolute_area_error_m2": error, "area_error_bound_m2": error_bound,
            "classification_disagreements": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "inspection_inputs.json")
    parser.add_argument("--output", type=Path, default=HERE / "fixture_tolerance_results.json")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    s = json.loads(raw)["plate"]
    rows = [analyze(s, error, error) for error in EQUAL_HALF_WIDTHS_M]
    nominal = rows[0]["maximin"]
    if nominal is None:
        raise ValueError("the nominal fixture cannot produce the required image")
    shape = (max(1, s["grid_distance_cells"] // 2), max(1, s["grid_offset_cells"] // 2))
    for row in rows:
        row["grid"] = check_grid(s, row, shape)
        bounds = row["d_bounds_m"]
        row["nominal_centered_optimum_still_feasible"] = bool(
            bounds and bounds[0] <= nominal["d_command_m"] <= bounds[1])
    tangent = math.tan(math.radians(s["horizontal_fov_deg"]) / 2)
    half = s["plate_width_m"] / 2
    cap = min(s["distance_max_m"], s["sensor_pixels"] * half / (tangent * s["min_plate_pixels"]))
    result = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "assumptions": "All combinations of bounded axial and lateral errors are possible. The optical axis remains normal to the plate. Bounds are illustrative, not measured.",
        "objective": "Maximize the worst fully projected width, then maximize the minimum pixel clearance on either side.",
        "fraction_denominator_m2": (s["distance_max_m"] - s["distance_min_m"]) * 2 * s["offset_limit_m"],
        "grid_definition": "Midpoint grid on the original mechanical rectangle, tested at all four physical error-box corners.",
        "largest_feasible_equal_half_width_m": min(s["offset_limit_m"], (cap - s["distance_min_m"]) / 2,
                                                   (cap * tangent - half) / (1 + 2 * tangent)),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['axial_half_width_m']:.3f} m: area={r['area_m2']:.8f} m², "
              f"grid error={r['grid']['absolute_area_error_m2']:.8g} m²")


if __name__ == "__main__":
    main()
