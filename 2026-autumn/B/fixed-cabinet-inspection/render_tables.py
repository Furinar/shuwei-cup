"""Rebuild the inspection paper's TeX tables from the measured JSON outputs."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EOL = r" \\"


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def safe_text(text):
    escapes = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return "".join(escapes.get(c, c) for c in text)


def interval(times):
    if times is None:
        return "---"
    start, end = times
    return f"{start:g}--{end:g}"


def review(segments):
    return r",\,".join(interval(segment) for segment in segments) if segments else "---"


def tabular(spec, head, rows, spanner=None):
    lines = [r"\begingroup\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}" + spec + r"@{}}",
             r"\toprule"]
    if spanner:
        lines.extend(spanner)
    lines += [" & ".join(head) + EOL, r"\midrule"]
    for row in rows:
        if len(row) != len(head):
            raise ValueError(f"row length {row!r} differs from header {head!r}")
        lines.append(" & ".join(map(str, row)) + EOL)
    lines += [r"\bottomrule", r"\end{tabular}\endgroup"]
    return "\n".join(lines)


def tables_for_inspection(inputs, results):
    plate = results["plate"]["by_threshold"]
    region = tabular("rrrrr", ["Threshold (px)", r"$d_{\max}$ (m)", r"Area (m$^2$)", "Fraction",
                                r"Grid (m$^2$)"],
                     [[r["min_pixels"], f'{r["distance_bounds_m"][1]:.4f}' if r["best"] else "---",
                       f'{r["area_m2"]:.6f}', f'{r["fraction"]:.4f}', f'{r["grid_area_m2"]:.6f}']
                      for r in plate])
    pairs = results["temperature"]["all_pairs"]
    budget = inputs["temperature"]["budget_instrument_min"]
    comparison = tabular("lrrrr", ["Pair", "Cost (min)", f"Fits {budget:g} min",
                                     r"Variance [$(^\circ\mathrm{C})^2$]", r"SE ($^\circ$C)"],
                         [["+".join(map(safe_text, r["pair"])), r["cost_min"],
                           "Yes" if r["feasible_at_nominal_budget"] else "No",
                           f'{r["variance_C2"]:.5f}', f'{r["standard_error_C"]:.5f}'] for r in pairs])
    return {"PlateTable": region, "PairTable": comparison}


def timeline_table(first, second, job_order, label1, label2):
    a = {r["job"]: r for r in first["jobs"]}
    b = {r["job"]: r for r in second["jobs"]}
    if set(job_order) != set(a) or set(a) != set(b):
        raise ValueError("the timeline tables require complete, identical job rosters")
    rows = [[safe_text(job), interval(a[job]["inspect"]), review(a[job]["review"]),
             interval(b[job]["inspect"]), review(b[job]["review"])] for job in job_order]
    spanner = [r" & \multicolumn{2}{c}{" + label1 + r"} & \multicolumn{2}{c}{" + label2 + "}" + EOL,
               r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"]
    return tabular("lcccc", ["Job", "Inspection", "Review", "Inspection", "Review"], rows, spanner)


def tables_for_schedule(inputs, results):
    roster = inputs["inspection_review_minutes"]
    base = inputs["initial_order"]
    jobs = tabular("lrrl", ["Job", "Inspection (min)", "Review (min)", "Release"],
                   [[safe_text(job), values[0], values[1], "Initial" if job in base else "At notice"]
                    for job, values in roster.items()])
    base_runs = results["q2"]
    baseline = timeline_table(base_runs["fcfs"], base_runs["optimal"], base, "FCFS", "Optimal order")
    cases = results["q4"]
    summary, audit = [], []
    for i, case in enumerate(cases):
        keep, alternative = case["keep"], case["replan"]
        ident = safe_text(case["id"])
        added = ", ".join(map(safe_text, case["added"])) or "---"
        change = alternative["k"] - keep["k"]
        summary.append([ident, added, case["review_outage_min"],
                        f'{keep["k"]}/{keep["N"]}', keep["makespan_min"],
                        f'{alternative["k"]}/{alternative["N"]}', alternative["makespan_min"],
                        f"{change:+d}" if change else "0"])
        if i and i % 2 == 0:
            audit.append(r"\clearpage")
        audit.append(r"\subsection*{" + ident + ". " + safe_text(case["label"]) + "}")
        close = case["review_outage_min"]
        closed = interval([inputs["event_min"], inputs["event_min"] + close])
        note = f"The desk closes during {closed} min." if close else "The desk remains open."
        for policy, record in (("Keep", keep), ("Reorder", alternative)):
            if record["setup"] is not None:
                note += f" {policy} changes the queue during {interval(record['setup'])} min."
        audit.append(note + r"\par")
        audit.append(r"\begin{center}\small")
        audit.append(timeline_table(keep, alternative, base + case["added"], "Keep", "Reorder"))
        audit.append(r"\end{center}")
    groups = [r" & & & \multicolumn{2}{c}{Keep} & \multicolumn{2}{c}{Reorder} & " + EOL,
              r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}"]
    scenario = tabular("llrrrrrr", ["Case", "New", "Outage (min)", "$k/N$", "Clear (min)", "$k/N$",
                                     "Clear (min)", r"$\Delta k$"], summary, groups)
    return {"JobsTable": jobs, "BaseTable": baseline, "ScenarioTable": scenario,
            "AuditAppendix": "\n".join(audit)}


def table_for_noise(results):
    rows = []
    for check in results["checks"]:
        mc = check["monte_carlo"]
        lo, hi = mc["wilson_95"]
        rows.append([f'{check["s_m"]:.3f}', f'{check["sigma_m"]:.4f}', f'{check["p_exact"]:.6f}',
                     f'{mc["rate"]:.6f}', f"[{lo:.6f}, {hi:.6f}]"])
    return tabular("rrrrc", ["$s$ (m)", r"$\sigma$ (m)", "Exact $p$", "MC rate", r"95\% Wilson interval"], rows)


def tolerance_table(results):
    rows = []
    for case in results["cases"]:
        axial = 100 * case["axial_half_width_m"]
        lateral = 100 * case["lateral_half_width_m"]
        best = case["maximin"]
        rows.append([f"{axial:g} / {lateral:g}", f'{100 * best["d_command_m"]:.1f}',
                     f'{100 * case["mechanical_area_fraction"]:.2f}',
                     f'{best["worst_width_px"]:.2f}', f'{best["worst_edge_clearance_px"]:.2f}'])
    return tabular("crrrr", ["Axial / lateral (cm)", r"$d_0$ (cm)", r"Area (\%)",
                             "Width (px)", "Margin (px)"], rows)


def recovery_table(inputs, results):
    case = results["cases"][inputs["figure_case"]]
    scores = case["profiles"][results["design_profile"]]["scores"]
    methods = [
        ("keep", "Keep queue"),
        ("short_total_work", "Shortest total work"),
        ("plug_in_mean", f"Assume {case['plug_in_minutes']:g} min"),
        ("distribution", "Optimize expected score"),
        ("perfect_information", "Known recovery time"),
    ]
    rows = []
    for key, label in methods:
        score = scores[key]
        order = case["fixed_orders"].get(key)
        rows.append([label, safe_text("".join(order)) if order else "Varies",
                     f'{score["expected_k"]:.3f}', f'{score["expected_clearing_min"]:.2f}'])
    return tabular("llrr", ["Decision rule", "Queue", r"$E[k]$", r"$E[C_{\max}]$ (min)"], rows)


def write_tradeoff_rows(frontier):
    weight = frontier["weight_sum"]
    rows = []
    for region in frontier["intervals"]:
        lo, hi = region["lower_v"], region["upper_v"]
        lower_relation = r"\le " if lo == "0" else "<"
        domain = f"{lo}{lower_relation}v" if hi is None else f"{lo}{lower_relation}v<{hi}"
        orders = ", ".join(safe_text("".join(order)) for order in region["orders"])
        rows.append(" & ".join([
            "$" + domain + "$", orders, f'{region["weighted_k_sum"] / weight:.2f}',
            f'{region["weighted_clearing_sum"] / weight:.2f}',
        ]) + EOL)
    macro = r"\newcommand{\TradeoffRows}{%" + "\n" + "\n".join(rows) + "%\n}\n"
    (HERE / "tradeoff_rows.tex").write_text(macro, encoding="utf-8")


def sensitivity_text(cases):
    by_minutes = {row["changeover_min"]: row for row in cases}
    selected = [by_minutes[c] for c in (0, 2, 10)]
    retained = selected[0]["keep"]
    if any(row["keep"] != retained for row in cases):
        raise ValueError("retained queue changes across S3 setup-cost experiments")
    changed = max((row for row in cases if row["queue_changed"]), key=lambda row: row["changeover_min"])
    unchanged = min((row for row in cases if row["changeover_min"] > changed["changeover_min"]
                     and not row["queue_changed"]), key=lambda row: row["changeover_min"])
    costs = [r["changeover_min"] for r in selected]
    counts = [r["replan"]["k"] for r in selected]
    clears = [r["replan"]["makespan_min"] for r in selected]
    return (f"In {safe_text(selected[0]['scenario'])} the original queue finishes {retained['k']} of "
            f"{retained['N']} orders on time and clears in {retained['makespan_min']} minutes. "
            f"With changeovers of {costs[0]}, {costs[1]} and {costs[2]} minutes, the on-time counts are "
            f"{counts[0]}, {counts[1]} and {counts[2]}. Clearance takes {clears[0]}, {clears[1]} and "
            f"{clears[2]} minutes, respectively. At {changed['changeover_min']} minutes reordering still "
            f"finishes {changed['replan']['k']} on time. At {unchanged['changeover_min']} minutes the "
            f"optimal policy retains the original queue.")


def main():
    inspection = read("inspection_results.json")
    input_hash = hashlib.sha256((HERE / "inspection_inputs.json").read_bytes()).hexdigest()
    if input_hash != inspection["input_sha256"]:
        raise ValueError("inspection results do not match current inspection inputs")
    macros = tables_for_inspection(read("inspection_inputs.json"), inspection)
    schedule = read("schedule_results.json")
    macros.update(tables_for_schedule(read("schedule_inputs.json"), schedule))
    macros["NoiseTable"] = table_for_noise(read("q3_results.json"))
    macros["SensitivityText"] = sensitivity_text(schedule["s3_changeover_sensitivity"])
    macros["ToleranceTable"] = tolerance_table(read("fixture_tolerance_results.json"))
    recovery_inputs = read("uncertain_inputs.json")
    recovery_results = read("uncertain_results.json")
    macros["RecoveryTable"] = recovery_table(recovery_inputs, recovery_results)
    write_tradeoff_rows(recovery_results["cases"][recovery_inputs["figure_case"]]["value_tradeoff"])
    parts = ["% Generated from the case inputs and computation results by render_tables.py."]
    for name, body in macros.items():
        parts.append("\\newcommand{\\" + name + "}{%\n" + body + "%\n}")
    (HERE / "case_tables.tex").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
