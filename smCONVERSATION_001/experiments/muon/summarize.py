"""Print aggregate tables from a Muon experiment result."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    document = json.loads(args.result.read_text())
    print("variant\tmean_best_val\tsd\tmean_bpb\tmedian_steps_per_second\tmean_clip_fraction")
    for variant in document["selected_variants"]:
        runs = [run for run in document["runs"] if run["variant"] == variant]
        losses = [run["best_validation_loss"] for run in runs]
        deviations = statistics.stdev(losses) if len(losses) > 1 else 0.0
        print("%s\t%.6f\t%.6f\t%.6f\t%.3f\t%.3f" % (
            variant,
            statistics.mean(losses),
            deviations,
            statistics.mean(run["best_bits_per_byte"] for run in runs),
            statistics.median(run["steps_per_second_including_eval"] for run in runs),
            statistics.mean(run["clipped_step_fraction"] for run in runs),
        ))


if __name__ == "__main__":
    main()
