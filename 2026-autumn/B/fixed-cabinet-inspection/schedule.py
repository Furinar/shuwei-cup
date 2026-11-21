"""Exact, offline scheduling of inspection records from known fixed cabinets.

Run with the project Python environment. All times are integer minutes.
"""

import argparse
import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def review_segments(ready, duration, outage_start, outage_duration):
    end = outage_start + outage_duration
    if not outage_duration or ready >= end or ready + duration <= outage_start:
        return [[ready, ready + duration]]
    if ready >= outage_start:
        return [[end, end + duration]]
    return [[ready, outage_start], [end, ready + duration + outage_duration]]


def simulate(order, times, event=0, frozen=0, changeover=0, outage=0):
    inspect_free = review_free = 0
    setup = None
    rows = []
    for i, job in enumerate(order):
        if i == frozen:
            inspect_free = max(event, inspect_free)
            if changeover:
                setup = [inspect_free, inspect_free + changeover]
                inspect_free += changeover
        p, q = times[job]
        inspect = [inspect_free, inspect_free + p]
        inspect_free = inspect[1]
        segments = review_segments(max(inspect_free, review_free), q, event, outage)
        review_free = segments[-1][1]
        rows.append({"job": job, "inspect": inspect, "review": segments})
    return {"order": list(order), "setup": setup, "jobs": rows,
            "makespan_min": review_free,
            "sum_completion_min": sum(row["review"][-1][1] for row in rows)}


def score(result, window):
    rows = result["jobs"]
    finished = sum(row["review"][-1][1] <= window for row in rows)
    result.update(k=finished, N=len(rows), eta=finished / len(rows),
                  window_tlast_min=result["makespan_min"] if finished == len(rows) else window)
    return result


def tick_check(result, times, event=0, frozen=0, changeover=0, outage=0):
    """Independent minute-step replay; a review can start once inspection ends."""
    order = result["order"]
    inspection_work = [[] for _ in order]
    review_work = [[] for _ in order]
    setup_work = []
    ix = rx = 0
    horizon = event + outage + changeover + sum(sum(times[job]) for job in order) + 1
    for minute in range(horizon):
        # The review desk sees the inspection completions from the preceding tick.
        if rx < ix and not event <= minute < event + outage:
            review_work[rx].append(minute)
            if len(review_work[rx]) == times[order[rx]][1]:
                rx += 1
        if ix < len(order) and (ix < frozen or minute >= event):
            if ix == frozen and len(setup_work) < changeover:
                setup_work.append(minute)
            else:
                inspection_work[ix].append(minute)
                if len(inspection_work[ix]) == times[order[ix]][0]:
                    ix += 1
        if rx == len(order):
            break
    assert len(setup_work) == changeover or frozen == len(order)
    if result["setup"]:
        assert setup_work == list(range(*result["setup"]))
    for i, row in enumerate(result["jobs"]):
        assert inspection_work[i] == list(range(*row["inspect"]))
        assert review_work[i] == [t for a, b in row["review"] for t in range(a, b)]


def validate_inputs(data):
    times = data["inspection_review_minutes"]
    initial = data["initial_order"]
    if not initial or len(set(initial)) != len(initial) or not set(initial) <= set(times):
        raise ValueError("initial_order must contain distinct known job IDs")
    if any(len(row) != 2 or any(type(v) is not int or v <= 0 for v in row) for row in times.values()):
        raise ValueError("processing times must be positive integer pairs")
    if any(type(data[key]) is not int or data[key] < 0
           for key in ["event_min", "window_end_min", "changeover_min"]):
        raise ValueError("event, window and changeover must be nonnegative integers")
    if not data["scenarios"] or len({s["id"] for s in data["scenarios"]}) != len(data["scenarios"]):
        raise ValueError("require at least one scenario and distinct scenario IDs")
    for s in data["scenarios"]:
        added = s["added"]
        if set(added) & set(initial) or len(added) != len(set(added)) or not set(added) <= set(times):
            raise ValueError("new job IDs must be distinct, known and absent from initial_order")
        if type(s["review_outage_min"]) is not int or s["review_outage_min"] < 0:
            raise ValueError("outage must be a nonnegative integer")


def evaluate_scenario(scenario, best, prefix, times, event, window, changeover):
    keep = tuple(best["order"]) + tuple(scenario["added"])
    outage = scenario["review_outage_min"]
    frozen = len(prefix)
    candidates = []
    for suffix in itertools.permutations(keep[frozen:]):
        seq = prefix + suffix
        delay = changeover if seq != keep else 0
        rec = score(simulate(seq, times, event, frozen, delay, outage), window)
        tick_check(rec, times, event, frozen, delay, outage)
        for nominal_row, frozen_row in zip(best["jobs"][:frozen], rec["jobs"][:frozen]):
            assert nominal_row["inspect"] == frozen_row["inspect"]
        candidates.append(rec)
    retained = candidates[0]
    opt = min(candidates, key=lambda r: (-r["k"], r["makespan_min"], r["order"] != list(keep),
                                         r["sum_completion_min"], r["order"]))
    assert len(candidates) == math.factorial(len(keep) - frozen)
    assert (-opt["k"], opt["makespan_min"]) <= (-retained["k"], retained["makespan_min"])
    tied = sum((r["k"], r["makespan_min"]) == (opt["k"], opt["makespan_min"]) for r in candidates)
    return {**scenario, "candidate_count": len(candidates), "score_optima_count": tied,
            "keep": retained, "replan": opt}


def solve(data):
    validate_inputs(data)
    times = data["inspection_review_minutes"]
    initial = data["initial_order"]
    event, window = data["event_min"], data["window_end_min"]
    changeover = data["changeover_min"]
    nominal = []
    for seq in itertools.permutations(initial):
        record = simulate(seq, times)
        tick_check(record, times)
        nominal.append(record)
    best = min(nominal, key=lambda r: (r["makespan_min"], r["sum_completion_min"], r["order"]))
    lower_bound = sum(times[j][0] for j in initial) + min(times[j][1] for j in initial)
    count = sum(r["makespan_min"] == best["makespan_min"] for r in nominal)
    nominal_order = tuple(best["order"])
    frozen = sum(r["inspect"][0] < event for r in best["jobs"])
    prefix = nominal_order[:frozen]
    early = sorted((j for j in initial if times[j][0] <= times[j][1]), key=lambda j: times[j][0])
    late = sorted((j for j in initial if times[j][0] > times[j][1]), key=lambda j: -times[j][1])
    johnson = simulate(early + late, times)
    assert johnson["makespan_min"] == best["makespan_min"]
    q2 = {"fcfs": nominal[0], "optimal": best, "johnson": johnson, "candidate_count": len(nominal),
          "makespan_optima_count": count, "lower_bound_min": lower_bound}

    scenarios = [evaluate_scenario(s, best, prefix, times, event, window, changeover)
                 for s in data["scenarios"]]
    sensitivity = []
    if config := data.get("changeover_sensitivity"):
        source = next((s for s in data["scenarios"] if s["id"] == config["scenario"]), None)
        if source is None:
            raise ValueError("unknown sensitivity scenario")
        for cost in config["minutes"]:
            if type(cost) is not int or cost < 0:
                raise ValueError("sensitivity costs must be nonnegative integer minutes")
            case = evaluate_scenario(source, best, prefix, times, event, window, cost)
            summaries = {name: {k: case[name][k] for k in ("order", "k", "N", "makespan_min")}
                         for name in ("keep", "replan")}
            sensitivity.append({"scenario": source["id"], "changeover_min": cost, **summaries,
                                "queue_changed": case["keep"]["order"] != case["replan"]["order"],
                                "score_optima_count": case["score_optima_count"],
                                "candidate_count": case["candidate_count"]})

    return {"q2": q2, "frozen_jobs": list(prefix), "q4": scenarios, "s3_changeover_sensitivity": sensitivity,
            "validation": {"independent_method": "minute-step replay of every resource interval in every candidate",
                           "nominal_candidates_checked": len(nominal),
                           "scenario_candidates_checked": sum(s["candidate_count"] for s in scenarios),
                           "sensitivity_candidates_checked": sum(s["candidate_count"] for s in sensitivity)}}


def plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blue, green = "#275681", "#13816e"
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.8), sharey=True, layout="constrained")
    rows = records["q4"]
    for ax, field, title in zip(axes, ("eta", "makespan_min"),
                                ("Finished by deadline (%)", "All records cleared (min)")):
        for y, case in enumerate(rows):
            a, b = case["keep"][field], case["replan"][field]
            if field == "eta":
                a, b = 100 * a, 100 * b
            ax.plot([a, b], [y, y], color="#a8b4bc", linewidth=2, zorder=1)
            ax.plot(a, y, "o", color=blue, mfc="white", mew=1.7, ms=8, zorder=2)
            ax.plot(b, y, "D", color=green, ms=4, zorder=3)
        ax.set_xlabel(title)
        ax.set_yticks(range(len(rows)), [s["id"] for s in rows])
        ax.grid(axis="x", alpha=.18)
        ax.spines[["right", "top"]].set_visible(False)
        ax.margins(x=.14, y=.14)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Scenario")
    handles = [Line2D([], [], marker="o", mfc="white", mec=blue, mew=1.7, ls="", label="Keep queue"),
               Line2D([], [], marker="D", color=green, ls="", label="Allow reordering")]
    fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "schedule_inputs.json")
    ap.add_argument("--results", type=Path, default=HERE / "schedule_results.json")
    ap.add_argument("--figure", type=Path, default=HERE / "schedule_comparison.pdf")
    args = ap.parse_args()
    result = solve(json.loads(args.input.read_text(encoding="utf-8")))
    args.results.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(result, args.figure)
    print(result["validation"])


if __name__ == "__main__":
    main()
