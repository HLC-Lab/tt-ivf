#!/usr/bin/env python3
"""Compare the per-implementation analyses written by run_energy_comparison.sh
per recall level.

Reads <root>/levels.txt, <root>/<implementation>/analysis[/searches]/
tt_energy_configuration_summary.csv (written by the unchanged ann_ivf_energy
analyzer) and the TTAISS recall runs (recall_ttaiss_<nlist>_<nprobe>.log,
outside the measured windows). Prints per level and implementation the median
system (TT board + CPU package) energy for all searches of a session, its
first and third quartiles over sessions, mJ per query, the median duration and
the TT/host split. "Full" includes device/index creation; "searches" sums only
the search intervals. Board + CPU package energy is not wall-socket energy; no
idle baseline is subtracted.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# batch/gather[/opt]; the opt part was added later, so older roots have two parts.
LEVEL = re.compile(r"^(\d+):(\d+)/(\d+):(\d+)(:rerank)?(?:/(\d+):(\d+)(:rerank)?)?$")
# "gather_fast" and "cluster_major_opt" are the names used by earlier roots.
IMPLEMENTATIONS = ("batch_major", "ttaiss", "gather_fast", "cluster_major_opt")


def load(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        return {(int(r["NList"]), int(r["NProbe"])): r for r in csv.DictReader(handle)}


def recall(root: Path, name: str, nlist: int, nprobe: int) -> str:
    log = root / f"recall_{name}_{nlist}_{nprobe}.log"
    if not log.exists():
        return "—"
    found = re.findall(r"Recall@10=([0-9.]+)", log.read_text(errors="replace"))
    return found[-1] if found else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    levels = [LEVEL.match(line.strip()) for line in (args.root / "levels.txt").read_text().splitlines() if line.strip()]
    if not all(levels):
        raise SystemExit("levels.txt has a malformed level")
    for window, sub in (("searches only", "analysis/searches"), ("full (initialization + searches)", "analysis")):
        data = {name: load(args.root / name / sub / "tt_energy_configuration_summary.csv")
                for name in IMPLEMENTATIONS if (args.root / name / sub / "tt_energy_configuration_summary.csv").exists()}
        print(f"### Window: {window}\n")
        print("| Level (batch / TTAISS) | Implementation | Config | Recall@10 | Sessions | System J (median) | Q1 J | Q3 J "
              "| System mJ/query | TT mJ/query | Duration s | TT W | Host W | vs batch-major |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for match in levels:
            parts = {"batch_major": ((int(match[1]), int(match[2])), False),
                     ("ttaiss" if "ttaiss" in data else "gather_fast"): ((int(match[3]), int(match[4])), bool(match[5]))}
            if match[6]:
                parts["cluster_major_opt"] = ((int(match[6]), int(match[7])), bool(match[8]))
            base = data.get("batch_major", {}).get(parts["batch_major"][0])
            for name, (key, reranked) in parts.items():
                r = data.get(name, {}).get(key)
                if r is None:
                    continue
                config = f"{key[0]}/{key[1]}" + (" + rerank" if reranked else "")
                level_recall = recall(args.root, name, *key) if name != "batch_major" else "—"
                total_mj_q = float(r["MedianTotalEnergyPerQuery_mJ"])
                ratio = (f"{total_mj_q / float(base['MedianTotalEnergyPerQuery_mJ']):.3f}x" if base is not None else "—")
                print(f"| {match[0]} | {name} | {config} | {level_recall} | {r['SessionCount']} "
                      f"| {float(r['MedianTotalEnergy_J']):.2f} | {float(r['Q1TotalEnergy_J']):.2f} "
                      f"| {float(r['Q3TotalEnergy_J']):.2f} | {total_mj_q:.3f} "
                      f"| {float(r['MedianTTEnergyPerQuery_mJ']):.3f} | {float(r['MedianDuration_s']):.3f} "
                      f"| {float(r['MedianTTAveragePower_W']):.2f} | {float(r['MedianHostAveragePower_W']):.2f} | {ratio} |")
        print()
    print("All columns are per-configuration medians or quartiles over sessions. ann_ivf_energy computes no recall; "
          "TTAISS recall comes from one ordinary run per level outside the measured windows.")


if __name__ == "__main__":
    main()
