"""用英文修订版的绘图函数和原始数值结果，另存五张中文矢量图。"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parent
sys.path.insert(0, str(PROJECT))

import inspection
import joint_decision
import q3_noise
import schedule
import uncertain_recovery


def read_result(stem):
    return json.loads((PROJECT / f"{stem}.json").read_text(encoding="utf-8"))


def main():
    plate = read_result("inspection_inputs")["plate"]
    plate_result = read_result("inspection_results")["plate"]
    joint = read_result("joint_results")
    angular = read_result("q3_results")
    repairs = read_result("uncertain_results")
    figure_case = read_result("uncertain_inputs")["figure_case"]

    threshold = plate_result["nominal_min_pixels"]
    se = joint["se_requirement_C"]
    labels = {
        "Normal distance from the plate, d (m)": "到铭牌的垂直距离 d（m）",
        "Lateral mounting offset, u (m)": "横向安装偏移 u（m）",
        f"Feasible at {threshold} pixels": f"合格区域（至少 {threshold} 像素）",
        "Centered optimum": "居中最优位置",
        "Best SE at or before deadline": "限时内可达的最小标准误",
        f"SE limit {se:.2f} °C": f"标准误上限 {se:.2f} °C",
        "Dominated pair": "被支配的通道组合",
        "Exceeds nominal budget": "超过标称预算",
        "All jobs cleared (min)": "全部工单清空时刻（min）",
        "Temperature standard error (°C)": "温度估计的标准误（°C）",
        "Exact sector integral": "扇区积分",
        "Direct simulation (95% CI)": "直接模拟（95% 置信区间）",
        "Calibration setting s (m)": "累计移动距离 s（m）",
        "Probability within ±5°": "角误差落在 ±5° 内的概率",
        "A. Angular coverage": "（a）角误差覆盖概率",
        "B. First-order approximation error": "（b）一阶近似的偏差",
        "Approximation minus exact (percentage points)": "近似概率减去积分概率（百分点）",
        "Finished by deadline (%)": "按期完成比例（%）",
        "All records cleared (min)": "全部工单清空时刻（min）",
        "Scenario": "情景",
        "Keep queue": "保留原序",
        "Allow reordering": "允许重排",
        "Decide at notice": "接到通知时决定队列",
        "Know repair duration": "预知维修时长",
        "Orders finished by the deadline": "按期完成工单数",
        "Time all orders clear (min)": "全部工单清空时刻（min）",
        "Actual repair duration (min)": "实际维修时长（min）",
    }
    families = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("STHeiti", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "FandolHei")
    cjk = next((family for family in candidates if family in families), None)
    if cjk is None:
        raise RuntimeError("请安装思源黑体或 FandolHei，再重新运行脚本。")

    # Only translate existing artists immediately before export. The English
    # functions retain control of curves, intervals, ticks and legend handles.
    original_savefig = Figure.savefig
    matched = set()

    def save_chinese(fig, path, **kwargs):
        for artist in fig.findobj(Text):
            label = artist.get_text()
            if label in labels:
                artist.set_text(labels[label])
                matched.add(label)
        kwargs.setdefault("metadata", {"CreationDate": None, "ModDate": None})
        return original_savefig(fig, path, **kwargs)

    rc = {"font.family": ["DejaVu Sans", cjk], "pdf.fonttype": 42}
    with matplotlib.rc_context(rc), patch.object(Figure, "savefig", save_chinese):
        inspection.plot_region(plate, plate_result, OUTPUT / "label_region.pdf")
        joint_decision.plot(joint, OUTPUT / "joint_frontier.pdf")
        q3_noise.make_plot(q3_noise.SectorIntegral(96), angular["s_limit_m"], angular["checks"],
                           OUTPUT / "q3_validation.pdf")
        schedule.plot(read_result("schedule_results"), OUTPUT / "schedule_comparison.pdf")
        uncertain_recovery.make_figure(
            repairs["cases"][figure_case], OUTPUT / "uncertain_recovery.pdf",
            short_note="维修耗时 4–5 分钟时，两条预先确定的队列都按期交付 {count} 单，保留原序可提前 {gain} 分钟清空队列。",
        )

    if missing := labels.keys() - matched:
        raise ValueError(f"原图文字有变动，请核对翻译：{sorted(missing)}")
    print(f"已用 {cjk} 重绘五张图，核对 {len(matched)} 项标注。")


if __name__ == "__main__":
    main()
