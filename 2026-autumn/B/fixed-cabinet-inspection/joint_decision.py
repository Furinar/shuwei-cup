"""Joint temperature calibration and workstation planning for the fixed cabinets.

One extension to the base case places a two-channel temperature check on order A.
Each channel's setup occupies the inspection bench, before that order is released
to review. The fixture and basic 11-minute inspection are otherwise unchanged.
"""

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from inspection import estimate_pair
from schedule import simulate


HERE = Path(__file__).resolve().parent
CALIBRATION_JOB = "A"
SERVICE_DEADLINES_MIN = (46, 48)
SE_REQUIREMENT_C = 0.37


def independent_makespan(order, times):
    """Two-stage critical path: inspection prefix and review suffix share job k."""
    inspection_prefix = 0
    review_suffix = sum(times[job][1] for job in order)
    longest = 0
    for job in order:
        inspection, review = times[job]
        inspection_prefix += inspection
        longest = max(longest, inspection_prefix + review_suffix)
        review_suffix -= review
    return longest


def schedule_pair(pair, thermal, initial_order, initial_times):
    estimate = estimate_pair(pair, thermal)
    times = dict(initial_times)
    inspection, review = times[CALIBRATION_JOB]
    times[CALIBRATION_JOB] = (inspection + estimate["cost_min"], review)
    candidates = []
    for permutation in itertools.permutations(initial_order):
        record = simulate(permutation, times)
        if record["makespan_min"] != independent_makespan(permutation, times):
            raise ArithmeticError(f"critical path disagrees for {estimate['pair']}, {permutation}")
        candidates.append(record)
    optimal = min(candidates, key=lambda r: (r["makespan_min"], r["sum_completion_min"], r["order"]))
    return {
        "channels": "+".join(estimate["pair"]),
        "setup_min": estimate["cost_min"],
        "variance_C2": estimate["variance_C2"],
        "se_C": estimate["standard_error_C"],
        "within_nominal_budget": estimate["feasible_at_nominal_budget"],
        "optimal_makespan_min": optimal["makespan_min"],
        "representative_order": optimal["order"],
        "registration_order_makespan_min": simulate(initial_order, times)["makespan_min"],
        "lower_bound_min": sum(times[job][0] for job in initial_order) + min(times[job][1] for job in initial_order),
        "makespan_optima_count": sum(r["makespan_min"] == optimal["makespan_min"] for r in candidates),
        "permutations_checked": len(candidates),
    }


def dominates(first, second):
    time1, error1 = first["optimal_makespan_min"], first["variance_C2"]
    time2, error2 = second["optimal_makespan_min"], second["variance_C2"]
    return time1 <= time2 and error1 <= error2 and (time1 < time2 or error1 < error2)


def summarize(thermal, schedule):
    initial = schedule["initial_order"]
    times = schedule["inspection_review_minutes"]
    if CALIBRATION_JOB not in initial:
        raise ValueError(f"initial orders must include the augmented order {CALIBRATION_JOB}")
    pairs = [schedule_pair(pair, thermal, initial, times)
             for pair in itertools.combinations(thermal["channels"], 2)]
    for pair in pairs:
        pair["dominated_by"] = [r["channels"] for r in pairs if dominates(r, pair)]
    frontier = sorted((p for p in pairs if not p["dominated_by"]),
                      key=lambda p: (p["optimal_makespan_min"], p["se_C"], p["channels"]))
    budgeted = [p for p in pairs if p["within_nominal_budget"]]
    deadline_choices = []
    for deadline in SERVICE_DEADLINES_MIN:
        feasible = [p for p in budgeted if p["optimal_makespan_min"] <= deadline]
        chosen = min(feasible, key=lambda p: (p["se_C"], p["optimal_makespan_min"], p["channels"]), default=None)
        deadline_choices.append({"deadline_min": deadline, "best_channels": chosen["channels"] if chosen else None})
    choices = [p for p in budgeted if p["se_C"] <= SE_REQUIREMENT_C]
    fastest = min(choices, key=lambda p: (p["optimal_makespan_min"], p["se_C"], p["channels"]), default=None)
    return {
        "assumption": "Only order A receives the two-channel check. Both setup times add serially to its basic inspection; none overlap. Channel IDs do not denote order IDs.",
        "non_dominance_scope": "All channel pairs before imposing the nominal instrument-minute budget. Minimize variance and clearing time. No mixing of discrete choices.",
        "inputs": ["inspection_inputs.json", "schedule_inputs.json"],
        "nominal_budget_instrument_min": thermal["budget_instrument_min"],
        "joint_decisions_checked": sum(p["permutations_checked"] for p in pairs),
        "independent_check": "Each recurrence makespan verified against the maximum inspection-prefix plus review-suffix sum.",
        "pairs": pairs,
        "unrestricted_frontier": [p["channels"] for p in frontier],
        "best_precision_with_nominal_budget_and_deadline": deadline_choices,
        "se_requirement_C": SE_REQUIREMENT_C,
        "fastest_with_nominal_budget_and_se_requirement": (
            {"channels": fastest["channels"], "makespan_min": fastest["optimal_makespan_min"]} if fastest else None
        ),
    }


def plot(result, path):
    pairs = sorted(result["pairs"], key=lambda p: p["optimal_makespan_min"])
    fig, ax = plt.subplots(figsize=(7.1, 3.35), layout="constrained")
    best_so_far = []
    for pair in pairs:
        previous = best_so_far[-1] if best_so_far else float("inf")
        best_so_far.append(min(previous, pair["se_C"]))
    times = [p["optimal_makespan_min"] for p in pairs]
    ax.step(times, best_so_far, where="post", lw=1.25, color="#698482",
            label="Best SE at or before deadline")
    for p in pairs:
        x, y = p["optimal_makespan_min"], p["se_C"]
        color = "#a25929" if p["dominated_by"] else "#135766"
        symbol = "X" if p["dominated_by"] else "o"
        fill = color if p["within_nominal_budget"] else "white"
        ax.scatter(x, y, marker=symbol, s=43, facecolor=fill, edgecolor=color, zorder=3)
        label_dy = -11 if 0 <= result["se_requirement_C"] - y < .012 else 5
        ax.annotate(p["channels"], (x, y), xytext=(5, label_dy), textcoords="offset points", fontsize=9)
    ax.axhline(result["se_requirement_C"], color="#7e7586", ls=":", lw=1.2,
               label=f"SE limit {result['se_requirement_C']:.2f} °C")
    ax.set(xlabel="All jobs cleared (min)", ylabel="Temperature standard error (°C)")
    ax.set_xticks(times)
    ax.margins(x=.09, y=.19)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], marker="X", color="#a25929", linestyle="none", label="Dominated pair"),
        Line2D([], [], marker="o", color="#135766", markerfacecolor="white", linestyle="none",
               label="Exceeds nominal budget"),
    ])
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    thermal = json.loads((HERE / "inspection_inputs.json").read_text())["temperature"]
    schedule = json.loads((HERE / "schedule_inputs.json").read_text())
    result = summarize(thermal, schedule)
    (HERE / "joint_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, HERE / "joint_frontier.pdf")
    print(f"Verified {result['joint_decisions_checked']} pair/order combinations")


if __name__ == "__main__":
    main()
