# 固定机柜巡检的测量公差与工单排程决策

以 2026 年数维杯秋季赛 B 题为背景的个人建模练习。本文另行设计了固定机柜的铭牌成像、测温与检修工单，角度标定沿用了原题第三问的部分参数。算例输入、推导、数值结果与中英文论文一并提供。

[阅读中文论文](中文版/固定设施巡检_修订版_中文.pdf) · [Read the English paper](paper.pdf) · [四问解读](中文解读.md)

中英文论文各 18 页。第 14 页的图 5 比较了维修时长对交付的影响，表 10 给出不同时间成本下的三种最优队列。

## 复现

在仓库根目录使用 Python 3.12。核对已发布的输入与输出可以直接运行：

```sh
python3 reproduce.py --verify
```

要重新计算并编译英文论文，请在 macOS 或 Linux 上安装 XeLaTeX、latexmk、Times New Roman 和模板使用的宏包，再运行：

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

中文版沿用相同计算结果，五张中文矢量图已随仓库提供。运行下列命令生成 `中文版/paper_zh.pdf`。

```sh
cd 中文版
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper_zh.tex
```

中文字体与重绘命令见 [中文版编译说明](中文版/编译说明.md)。发布清单采用可移植路径，保留了原始算例、程序与输出的 SHA-256 校验值。
