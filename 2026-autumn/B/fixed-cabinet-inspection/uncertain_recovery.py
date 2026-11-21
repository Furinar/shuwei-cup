"""Evaluate one-time inspection queue decisions made before a repair ends.

All expectations are exact finite sums over constructed duration distributions.
The schedule and the independent minute-step replay use the case's shared rules.
"""

import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
from schedule import score, simulate, tick_check, validate_inputs


ROOT = Path(__file__).resolve().parent


def inputs():
    data = json.loads((ROOT / "schedule_inputs.json").read_text(encoding="utf-8"))
    design = json.loads((ROOT / "uncertain_inputs.json").read_text(encoding="utf-8"))
    validate_inputs(data)
    durations = design["duration_minutes"]
    if not durations or sorted(set(durations)) != durations or any(type(x) is not int or x < 1 for x in durations):
        raise ValueError("repair durations must be increasing, distinct positive integers")
    for weights in design["weight_profiles"].values():
        if len(weights) != len(durations) or any(type(w) is not int or w <= 0 for w in weights):
            raise ValueError("each duration requires a positive integer weight")
    if design["design_profile"] not in design["weight_profiles"]:
        raise ValueError("missing design profile")
    if len(set(design["conditional_cases"])) != len(design["conditional_cases"]):
        raise ValueError("duplicate conditional case")
    if design["figure_case"] not in design["conditional_cases"]:
        raise ValueError("figure case must be an evaluated case")
    return data, design


def weighted_totals(records, weights):
    # Integral numerators avoid tolerances or premature rounding in a tie.
    return tuple(sum(record[key] * w for record, w in zip(records, weights, strict=True))
                 for key in ("k", "makespan_min", "sum_completion_min"))


def best_orders(outcomes, weights, keep):
    def key(order):
        count, clearing, flow = weighted_totals(outcomes[order], weights)
        return -count, clearing, order != keep, flow, order

    order = min(outcomes, key=key)
    objective = key(order)[:2]
    tied = sum(key(candidate)[:2] == objective for candidate in outcomes)
    return order, tied


def schedule_choices(data, design, case, nominal):
    times = data["inspection_review_minutes"]
    event = data["event_min"]
    prefix = tuple(j["job"] for j in nominal["jobs"] if j["inspect"][0] < event)
    keep = tuple(nominal["order"] + case["added"])
    durations = design["duration_minutes"]
    baseline = simulate(nominal["order"], times)
    observed = [j["inspect"] for j in baseline["jobs"][:len(prefix)]]
    outcomes = {}
    for suffix in itertools.permutations(keep[len(prefix):]):
        order = prefix + suffix
        change = data["changeover_min"] if order != keep else 0
        by_duration = []
        for duration in durations:
            record = score(simulate(order, times, event, len(prefix), change, duration), data["window_end_min"])
            tick_check(record, times, event, len(prefix), change, duration)
            assert [j["inspect"] for j in record["jobs"][:len(prefix)]] == observed
            if record["setup"]:
                assert record["setup"][0] >= event
            by_duration.append({key: record[key] for key in ("k", "makespan_min", "sum_completion_min")})
        outcomes[order] = by_duration
    assert len(outcomes) == math.factorial(len(keep) - len(prefix))
    return prefix, keep, outcomes


def summarize(records, weights, n_jobs):
    k, clear, _ = weighted_totals(records, weights)
    total = sum(weights)
    return {"expected_k": k / total, "expected_fraction": k / (total * n_jobs),
            "expected_clearing_min": clear / total,
            "weighted_k_sum": k, "weighted_clearing_sum": clear, "weight_sum": total}


def value_tradeoff(outcomes, weights):
    """Minimize E[C] - v E[k] for every nonnegative value v (minutes per order)."""
    groups = {}
    for order, records in sorted(outcomes.items()):
        k, clearing, _ = weighted_totals(records, weights)
        groups.setdefault((k, clearing), []).append(list(order))

    # For each cost line, intersect its pairwise optimality inequalities with
    # v >= 0. Common weight denominators cancel, so all endpoints are exact.
    intervals = []
    endpoints = {Fraction(0)}
    for (k, clearing), orders in groups.items():
        lower, upper = Fraction(0), None
        for other_k, other_clearing in groups:
            if k == other_k:
                if clearing > other_clearing:
                    break
                continue
            crossing = Fraction(clearing - other_clearing, k - other_k)
            if k > other_k:
                lower = max(lower, crossing)
            else:
                upper = crossing if upper is None else min(upper, crossing)
        else:
            if upper is not None and lower > upper:
                continue
            endpoints.add(lower)
            if upper is not None:
                endpoints.add(upper)
            if lower != upper:
                intervals.append((lower, upper, k, clearing, orders))

    # Evaluate boundary costs separately to retain all ties, including queues
    # which are optimal only at one point. Each order is fixed across durations.
    def winners(value):
        costs = {(k, clearing): clearing - value * k for k, clearing in groups}
        best = min(costs.values())
        return sorted(order for line in groups if costs[line] == best for order in groups[line])

    rows = []
    for lower, upper, k, clearing, orders in sorted(intervals, key=lambda row: row[0]):
        probe = lower + 1 if upper is None else (lower + upper) / 2
        assert winners(probe) == orders
        rows.append({
            "lower_v": str(lower), "upper_v": None if upper is None else str(upper),
            "weighted_k_sum": k, "weighted_clearing_sum": clearing, "orders": orders,
        })
    return {
        "weight_sum": sum(weights),
        "intervals": rows,
        "boundaries": [{"v": str(v), "orders": winners(v)} for v in sorted(endpoints)],
    }


def experiment(data, design, case, nominal):
    durations = design["duration_minutes"]
    profiles = design["weight_profiles"]
    nominal_weights = profiles[design["design_profile"]]
    prefix, keep, outcomes = schedule_choices(data, design, case, nominal)
    # The plug-in policy solves for the single support value closest to the
    # prior mean (choose the lower on an exact tie), before seeing the repair.
    numerator = sum(d * w for d, w in zip(durations, nominal_weights))
    plug_index = min(range(len(durations)), key=lambda i: abs(durations[i] * sum(nominal_weights) - numerator))
    perfect_orders = []
    for i in range(len(durations)):
        delta = [int(i == j) for j in range(len(durations))]
        perfect_orders.append(best_orders(outcomes, delta, keep)[0])
    oracle = [outcomes[order][i] for i, order in enumerate(perfect_orders)]
    selected, ties = best_orders(outcomes, nominal_weights, keep)
    suffix = sorted(keep[len(prefix):], key=lambda j: (sum(data["inspection_review_minutes"][j]), j))
    fixed = {"keep": keep, "short_total_work": prefix + tuple(suffix),
             "plug_in_mean": perfect_orders[plug_index], "distribution": selected}
    # Each implementable plan below is one tuple for every possible actual duration.
    assert all(order in outcomes and order[:len(prefix)] == prefix for order in fixed.values())
    strategies = {name: outcomes[order] for name, order in fixed.items()}
    strategies["perfect_information"] = oracle
    for i, benchmark in enumerate(oracle):
        for rows in outcomes.values():
            assert (-benchmark["k"], benchmark["makespan_min"]) <= (-rows[i]["k"], rows[i]["makespan_min"])
    evaluations = {}
    for name, weights in profiles.items():
        optimum, n_ties = best_orders(outcomes, weights, keep)
        metrics = {pid: summarize(rows, weights, len(keep)) for pid, rows in strategies.items()}
        metrics["refit_for_profile"] = summarize(outcomes[optimum], weights, len(keep))
        evaluations[name] = {
            "mean_repair_min": sum(d*w for d, w in zip(durations, weights)) / sum(weights),
            "refit_order": list(optimum), "refit_ties_on_primary_and_secondary": n_ties,
            "scores": metrics,
        }
    result = {
        "case": case["id"], "new_jobs": case["added"], "N": len(keep), "frozen": list(prefix),
        "candidate_orders": len(outcomes), "minute_replays": len(outcomes)*len(durations),
        "nominal_ties_on_primary_and_secondary": ties, "plug_in_minutes": durations[plug_index],
        "fixed_orders": {name: list(order) for name, order in fixed.items()},
        "profiles": evaluations,
        "by_realized_duration": [
            {"duration_min": d, "oracle_order": list(perfect_orders[i]),
             "outcomes": {name: {key: rows[i][key] for key in ("k", "makespan_min")}
                          for name, rows in strategies.items()}}
            for i, d in enumerate(durations)
        ],
    }
    if case["id"] == design["figure_case"]:
        result["value_tradeoff"] = value_tradeoff(outcomes, nominal_weights)
    return result


def make_figure(
    case, path,
    short_note=("For 4–5 min repairs, both fixed queues deliver {count} orders on time. "
                "Keeping the queue clears all orders {gain} min earlier."),
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 3.4), layout="constrained")
    rows = case["by_realized_duration"]
    x = [row["duration_min"] for row in rows]
    labels = {"keep": "Keep queue", "distribution": "Decide at notice",
              "perfect_information": "Know repair duration"}
    styles = {"keep": dict(color="#516474", marker="o", mfc="white", ls="--"),
              "distribution": dict(color="#af4c2c", marker="D", ls="-"),
              "perfect_information": dict(color="#17675e", marker="+", ls=":")}
    for method in labels:
        for ax, key in ((left, "k"), (right, "makespan_min")):
            y = [row["outcomes"][method][key] for row in rows]
            ax.plot(x, y, ms=4, lw=1.35, label=labels[method], **styles[method])
    left.set_ylabel("Orders finished by the deadline")
    left.yaxis.get_major_locator().set_params(integer=True)
    right.set_ylabel("Time all orders clear (min)")
    for ax in (left, right):
        ax.set_xlabel("Actual repair duration (min)")
        ax.set_xticks(x[::2])
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
    short_repairs = [row["outcomes"] for row in rows if row["duration_min"] in (4, 5)]
    differences = {
        (result["keep"]["k"], result["distribution"]["k"],
         result["distribution"]["makespan_min"] - result["keep"]["makespan_min"])
        for result in short_repairs
    }
    if len(short_repairs) == 2 and len(differences) == 1:
        keep_count, revised_count, gain = differences.pop()
        if keep_count == revised_count and gain > 0:
            for ax in (left, right):
                ax.axvspan(4, 5, facecolor="#f3deb3", alpha=.65, zorder=0)
                ax.set_xticks(sorted(set(x[::2]) | {4, 5}))
            fig.supxlabel(short_note.format(count=keep_count, gain=gain),
                          fontsize=9, color="#533b23")
    handles, legend = left.get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside upper center", ncols=3, frameon=False, fontsize=9)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    data, design = inputs()
    times = data["inspection_review_minutes"]
    schedules = (simulate(order, times) for order in itertools.permutations(data["initial_order"]))
    nominal = min(schedules, key=lambda r: (r["makespan_min"], r["sum_completion_min"], r["order"]))
    source = {case["id"]: case for case in data["scenarios"]}
    cases = {}
    for case_id in design["conditional_cases"]:
        if case_id not in source or source[case_id]["review_outage_min"] <= 0:
            raise ValueError("conditional case must correspond to a closure in schedule_inputs.json")
        cases[case_id] = experiment(data, design, source[case_id], nominal)
    result = {
        "support_min": design["duration_minutes"], "weight_profiles": design["weight_profiles"],
        "design_profile": design["design_profile"], "initial_queue": nominal["order"],
        "evaluation": "Integer-weighted exact means. For each observed roster, select a fixed suffix by expected timely count then expected actual clearance; ties prefer keep, smaller total completion time then job IDs. Tilt scores re-use the nominal plan and report the tilt-specific optimum separately.",
        "cases": cases,
        "candidate_duration_replays": sum(case["minute_replays"] for case in cases.values()),
    }
    (ROOT / "uncertain_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figure(cases[design["figure_case"]], ROOT / "uncertain_recovery.pdf")
    print(f"PASS: {result['candidate_duration_replays']} schedule-duration pairs verified by minute replay.")


if __name__ == "__main__":
    main()
