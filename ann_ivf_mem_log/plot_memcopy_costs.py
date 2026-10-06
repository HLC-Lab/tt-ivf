#!/usr/bin/env python3

"""Plot stacked std::chrono stage timings across ANN IVF workloads."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:
    plt = None
    np = None
    PLOTTING_IMPORT_ERROR = error
else:
    PLOTTING_IMPORT_ERROR = None


STAGES = [
    ("h2d", "T_H2D", "#4C78A8"),
    ("cpu_coarse_config", "T_CPU,coarse", "#F58518"),
    ("tt_coarse_search", "T_TT,coarse", "#54A24B"),
    ("cpu_fine_prep", "T_CPU,fine", "#17BECF"),
    ("tt_fine_search", "T_TT,fine", "#B279A2"),
    ("cpu_output", "T_CPU,output", "#E45756"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create one stacked bar per nprobe using only chrono_stage rows " "written by ann_ivf_mem_log.")
    )
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument(
        "--mem-log",
        action="append",
        required=True,
        metavar="NPROBE=PATH",
        help="Repeat once per nprobe, for example --mem-log 8=logs/nprobe_8_mem_log.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ann_ivf_memcopy_costs.png"),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        help="CSV behind the plot; defaults to the PNG path with a .csv suffix",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Interactive HTML; defaults to the PNG path with an .html suffix",
    )
    parser.add_argument(
        "--without-tt-fine-output",
        type=Path,
        help=("Detail PNG with TT fine search omitted; defaults to " "<output-stem>_without_tt_fine.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_path_map(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--mem-log expects NPROBE=PATH, received {value!r}")
        nprobe_text, path_text = value.split("=", 1)
        nprobe = int(nprobe_text)
        if nprobe in result:
            raise ValueError(f"Duplicate nprobe={nprobe}")
        result[nprobe] = Path(path_text)
    return result


def measured_run_number(context: str) -> int | None:
    if context.startswith("run_") and context[4:].isdigit():
        return int(context[4:])
    return None


def read_stage_medians(path: Path) -> tuple[dict[str, float], int]:
    per_run: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"context", "layer", "operation", "duration_us"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            if row["layer"].strip() != "chrono_stage":
                continue
            run_number = measured_run_number(row["context"].strip())
            if run_number is None:
                continue
            operation = row["operation"].strip()
            if operation not in {stage[0] for stage in STAGES}:
                continue
            per_run[run_number][operation] = float(row["duration_us"])

    expected = {stage[0] for stage in STAGES}
    complete_runs = [stages for _, stages in sorted(per_run.items()) if set(stages) == expected]
    if not complete_runs:
        raise ValueError(
            f"{path} has no complete measured run with all six chrono_stage rows. "
            "Rebuild ann_ivf_mem_log after updating the C++ instrumentation."
        )

    medians = {
        operation: statistics.median(run_stages[operation] for run_stages in complete_runs) for operation in expected
    }
    return medians, len(complete_runs)


def collect_rows(nlist: int, mem_logs: dict[int, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for nprobe in sorted(mem_logs):
        medians, run_count = read_stage_medians(mem_logs[nprobe])
        row: dict[str, object] = {
            "nlist": nlist,
            "nprobe": nprobe,
            "measured_runs": run_count,
        }
        for operation, _, _ in STAGES:
            row[f"{operation}_median_us"] = medians[operation]
        row["stack_total_median_us"] = sum(medians.values())
        rows.append(row)
    return rows


def write_plot_data(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in row.items():
                if key.endswith("_us"):
                    formatted[key] = f"{float(value):.6f}"
            writer.writerow(formatted)


def plot_rows(
    rows: list[dict[str, object]],
    output: Path,
    dpi: int,
    excluded_operations: set[str] | None = None,
) -> None:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "plot_memcopy_costs.py requires NumPy and Matplotlib in the active "
            f"Python environment: {PLOTTING_IMPORT_ERROR}"
        )
    assert plt is not None
    assert np is not None

    excluded = excluded_operations or set()
    visible_stages = [stage for stage in STAGES if stage[0] not in excluded]
    if not visible_stages:
        raise ValueError("At least one stage must remain visible")

    nprobes = np.array([int(row["nprobe"]) for row in rows])
    x = np.arange(len(nprobes), dtype=float)
    bottom = np.zeros(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(11.2, 6.4), constrained_layout=True)
    for operation, label, color in visible_stages:
        values = np.array(
            [float(row[f"{operation}_median_us"]) for row in rows],
            dtype=float,
        ) / 1000.0
        ax.bar(
            x,
            values,
            width=0.62,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values

    totals = bottom.copy()
    label_bottom = np.zeros(len(rows), dtype=float)
    for operation, _, _ in visible_stages:
        values = np.array(
            [float(row[f"{operation}_median_us"]) for row in rows],
            dtype=float,
        ) / 1000.0
        for index, value in enumerate(values):
            if value <= 0:
                continue
            text = f"{value:.2f} ms"
            if value / totals[index] >= 0.055:
                ax.text(
                    x[index],
                    label_bottom[index] + value / 2.0,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#202020" if operation in {"cpu_coarse_config", "cpu_fine_prep"} else "white",
                    fontweight="bold",
                )
            elif "tt_fine_search" in excluded and operation in {"h2d", "cpu_output"}:
                ax.annotate(
                    text,
                    (x[index] + 0.32, label_bottom[index] + value / 2.0),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom" if operation == "h2d" else "center",
                    fontsize=8,
                    color="#202020",
                )
        label_bottom += values

    ax.set_xticks(x, [str(value) for value in nprobes])
    ax.set_xlim(-0.55, len(rows) - 0.05)
    ax.set_xlabel("N_probe")
    ax.set_ylabel("Median stage time (ms)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if "tt_fine_search" in excluded:
        total_prefix = "Visible subtotal"
    else:
        total_prefix = "Total"
    ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    vertical_padding = max(float(bottom.max()) * 0.015, 1.0)
    for index, total in enumerate(totals):
        ax.text(
            x[index],
            total + vertical_padding,
            f"{total_prefix}: {total:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, float(totals.max()) * 1.12)

    detail_note = (
        " TT fine search is intentionally omitted and totals are visible-stage subtotals."
        if "tt_fine_search" in excluded
        else ""
    )
    fig.text(
        0.5,
        0.01,
        (
            "All segments are disjoint std::chrono wall-clock stages. "
            "TT stages use enqueue + Finish; no Tracy/device-profiler data is used."
            f"{detail_note}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_interactive_html(path: Path, rows: list[dict[str, object]]) -> None:
    stage_definitions = [
        {
            "key": operation,
            "field": f"{operation}_median_us",
            "label": label,
            "color": color,
        }
        for operation, label, color in STAGES
    ]
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ANN IVF chrono stage breakdown — nlist=__NLIST__</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #172033;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; }
    .card {
      width: min(1180px, 100%);
      margin: 0 auto;
      padding: 24px 26px 18px;
      background: white;
      border: 1px solid #dfe5ee;
      border-radius: 14px;
      box-shadow: 0 8px 28px rgba(20, 36, 60, 0.09);
    }
    h1 { margin: 0; font-size: clamp(22px, 3vw, 32px); font-weight: 680; }
    .subtitle { margin: 7px 0 18px; color: #5b6577; font-size: 14px; }
    .chart-wrap { position: relative; min-height: 520px; }
    svg { display: block; width: 100%; height: auto; overflow: visible; }
    .axis-label { fill: #536074; font-size: 13px; }
    .tick-label { fill: #6a7485; font-size: 12px; }
    .segment-label { fill: white; font-size: 11px; font-weight: 700; pointer-events: none; }
    .total-label { fill: #1f2938; font-size: 12px; font-weight: 700; }
    .grid { stroke: #dfe5ec; stroke-width: 1; stroke-dasharray: 4 4; }
    .axis { stroke: #8d98a8; stroke-width: 1; }
    rect.segment { cursor: pointer; transition: opacity 100ms ease; }
    rect.segment:hover { opacity: 0.82; }
    .legend-background { fill: #f9fbfd; stroke: #d9e0e9; stroke-width: 1; }
    .legend-title { fill: #263247; font-size: 13px; font-weight: 700; }
    .legend-item { cursor: pointer; }
    .legend-item:hover { opacity: 0.76; }
    .legend-label { fill: #344055; font-size: 12px; }
    .legend-action { fill: #2868a9; font-size: 12px; font-weight: 650; cursor: pointer; }
    .legend-action:hover { text-decoration: underline; }
    .empty { fill: #7b8595; font-size: 16px; }
    .note { margin: 8px 0 0; color: #697486; font-size: 12px; text-align: center; }
    #tooltip {
      position: fixed;
      z-index: 10;
      display: none;
      pointer-events: none;
      padding: 8px 10px;
      border-radius: 8px;
      background: rgba(19, 27, 41, 0.94);
      color: white;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 5px 18px rgba(0, 0, 0, 0.2);
    }
    @media (max-width: 700px) {
      body { padding: 8px; }
      .card { padding: 16px 10px 12px; border-radius: 10px; }
      .chart-wrap { min-height: 420px; }
    }
  </style>
</head>
<body>
  <main class="card">
    <h1>ANN IVF chrono stage breakdown — nlist=__NLIST__</h1>
    <p class="subtitle">Click a stage in the legend inside the figure to hide or restore it.</p>
    <div class="chart-wrap">
      <svg id="chart" viewBox="0 0 1100 570" role="img" aria-label="Interactive stacked stage timing chart"></svg>
      <div id="tooltip"></div>
    </div>
    <p class="note">Medians across measured runs. Warmup is excluded. TT stages use enqueue + Finish; no device-profiler data is used.</p>
  </main>
  <script>
    const rows = __ROWS__;
    const stages = __STAGES__;
    const enabled = new Set(stages.map(stage => stage.key));
    const svg = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");
    const NS = "http://www.w3.org/2000/svg";

    function svgElement(name, attributes = {}, text = "") {
      const element = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      if (text) element.textContent = text;
      return element;
    }

    function formatDuration(us) {
      if (us >= 1000) return `${(us / 1000).toFixed(3)} ms`;
      return `${us.toFixed(2)} µs`;
    }

    function niceMaximum(value) {
      if (value <= 0) return 1;
      const magnitude = 10 ** Math.floor(Math.log10(value));
      const fraction = value / magnitude;
      const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
      return niceFraction * magnitude;
    }

    function makeInteractive(element, callback) {
      element.setAttribute("tabindex", "0");
      element.setAttribute("role", "button");
      element.addEventListener("click", callback);
      element.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          callback();
        }
      });
    }

    function drawLegend(width, margin) {
      const x = width - margin.right + 22;
      const y = margin.top;
      const legendWidth = margin.right - 46;
      const itemHeight = 29;
      const legendHeight = 50 + stages.length * itemHeight + 42;

      svg.appendChild(svgElement("rect", {
        x, y, width: legendWidth, height: legendHeight, rx: 9,
        class: "legend-background"
      }));
      svg.appendChild(svgElement("text", {
        x: x + 14, y: y + 24, class: "legend-title"
      }, "Timing stages (click to toggle)"));

      [...stages].reverse().forEach((stage, index) => {
        const visible = enabled.has(stage.key);
        const itemY = y + 46 + index * itemHeight;
        const group = svgElement("g", {
          class: "legend-item",
          opacity: visible ? "1" : "0.38",
          "aria-label": `${stage.label}: ${visible ? "visible" : "hidden"}`
        });
        group.appendChild(svgElement("rect", {
          x: x + 14, y: itemY - 12, width: 15, height: 15, rx: 3,
          fill: stage.color,
          stroke: visible ? "none" : "#596579",
          "stroke-width": visible ? "0" : "1"
        }));
        group.appendChild(svgElement("text", {
          x: x + 39, y: itemY, class: "legend-label",
          "text-decoration": visible ? "none" : "line-through"
        }, stage.label));
        makeInteractive(group, () => {
          visible ? enabled.delete(stage.key) : enabled.add(stage.key);
          render();
        });
        svg.appendChild(group);
      });

      const actionY = y + 57 + stages.length * itemHeight;
      const showAll = svgElement("text", {
        x: x + 14, y: actionY, class: "legend-action"
      }, "Show all");
      makeInteractive(showAll, () => {
        stages.forEach(stage => enabled.add(stage.key));
        render();
      });
      svg.appendChild(showAll);

      const hideAll = svgElement("text", {
        x: x + 87, y: actionY, class: "legend-action"
      }, "Hide all");
      makeInteractive(hideAll, () => {
        enabled.clear();
        render();
      });
      svg.appendChild(hideAll);
    }

    function render() {
      svg.replaceChildren();
      tooltip.style.display = "none";

      const width = 1100;
      const height = 570;
      const margin = {top: 30, right: 300, bottom: 72, left: 92};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const visibleStages = stages.filter(stage => enabled.has(stage.key));
      const totals = rows.map(row =>
        visibleStages.reduce((sum, stage) => sum + Number(row[stage.field]) / 1000, 0)
      );
      drawLegend(width, margin);

      if (visibleStages.length === 0) {
        svg.appendChild(svgElement("text", {
          x: margin.left + plotWidth / 2,
          y: height / 2,
          "text-anchor": "middle",
          class: "empty"
        }, "All stages are hidden. Click a legend entry to add it back."));
        return;
      }

      const yMaximum = niceMaximum(Math.max(...totals) * 1.08);
      const y = value => margin.top + plotHeight - (value / yMaximum) * plotHeight;
      const groupWidth = plotWidth / rows.length;
      const barWidth = Math.min(145, groupWidth * 0.56);

      for (let tick = 0; tick <= 5; tick += 1) {
        const value = yMaximum * tick / 5;
        const tickY = y(value);
        svg.appendChild(svgElement("line", {
          x1: margin.left, x2: width - margin.right, y1: tickY, y2: tickY, class: "grid"
        }));
        svg.appendChild(svgElement("text", {
          x: margin.left - 12, y: tickY + 4, "text-anchor": "end", class: "tick-label"
        }, `${value.toFixed(1)} ms`));
      }

      svg.appendChild(svgElement("line", {
        x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis"
      }));
      svg.appendChild(svgElement("line", {
        x1: margin.left, x2: width - margin.right,
        y1: height - margin.bottom, y2: height - margin.bottom, class: "axis"
      }));

      rows.forEach((row, rowIndex) => {
        const centerX = margin.left + groupWidth * (rowIndex + 0.5);
        let accumulated = 0;

        for (const stage of visibleStages) {
          const valueUs = Number(row[stage.field]);
          const value = valueUs / 1000;
          const top = y(accumulated + value);
          const bottom = y(accumulated);
          const segmentHeight = Math.max(0, bottom - top);
          const rect = svgElement("rect", {
            x: centerX - barWidth / 2,
            y: top,
            width: barWidth,
            height: segmentHeight,
            fill: stage.color,
            class: "segment"
          });
          rect.addEventListener("mousemove", event => {
            tooltip.innerHTML = `<strong>N_probe=${row.nprobe}</strong><br>${stage.label}: ${formatDuration(valueUs)}`;
            tooltip.style.display = "block";
            tooltip.style.left = `${event.clientX + 12}px`;
            tooltip.style.top = `${event.clientY + 12}px`;
          });
          rect.addEventListener("mouseleave", () => tooltip.style.display = "none");
          rect.addEventListener("click", () => {
            enabled.delete(stage.key);
            render();
          });
          svg.appendChild(rect);

          if (segmentHeight >= 28) {
            svg.appendChild(svgElement("text", {
              x: centerX,
              y: top + segmentHeight / 2 + 4,
              "text-anchor": "middle",
              class: "segment-label"
            }, `${value.toFixed(2)} ms`));
          }
          accumulated += value;
        }

        svg.appendChild(svgElement("text", {
          x: centerX,
          y: y(accumulated) - 9,
          "text-anchor": "middle",
          class: "total-label"
        }, `Total ${accumulated.toFixed(2)} ms`));
        svg.appendChild(svgElement("text", {
          x: centerX,
          y: height - margin.bottom + 25,
          "text-anchor": "middle",
          class: "tick-label"
        }, String(row.nprobe)));
      });

      svg.appendChild(svgElement("text", {
        x: margin.left + plotWidth / 2,
        y: height - 18,
        "text-anchor": "middle",
        class: "axis-label"
      }, "N_probe"));
      const yLabel = svgElement("text", {
        x: 20,
        y: margin.top + plotHeight / 2,
        "text-anchor": "middle",
        class: "axis-label",
        transform: `rotate(-90 20 ${margin.top + plotHeight / 2})`
      }, "Median stage time (ms)");
      svg.appendChild(yLabel);
    }

    render();
  </script>
</body>
</html>
"""
    document = document.replace("__NLIST__", str(rows[0]["nlist"]))
    document = document.replace("__ROWS__", json.dumps(rows, separators=(",", ":")))
    document = document.replace("__STAGES__", json.dumps(stage_definitions, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.nlist, parse_path_map(args.mem_log))
    data_output = args.data_output or args.output.with_suffix(".csv")
    html_output = args.html_output or args.output.with_suffix(".html")
    without_tt_fine_output = args.without_tt_fine_output or args.output.with_name(
        f"{args.output.stem}_without_tt_fine{args.output.suffix}"
    )
    write_plot_data(data_output, rows)
    plot_rows(rows, args.output, args.dpi)
    plot_rows(
        rows,
        without_tt_fine_output,
        args.dpi,
        excluded_operations={"tt_fine_search"},
    )
    write_interactive_html(html_output, rows)
    print(f"Wrote stacked chrono plot to {args.output}")
    print(f"Wrote detail plot without TT fine search to {without_tt_fine_output}")
    print(f"Wrote plotted stage medians to {data_output}")
    print(f"Wrote interactive stage chart to {html_output}")


if __name__ == "__main__":
    main()
