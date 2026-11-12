from __future__ import annotations

import argparse
import json

from lucidbench.annotation_sampling import create_annotation_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local, reproducible v0.3 annotation sample."
    )
    parser.add_argument("--config", default="configs/v0.3.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = create_annotation_sample(args.config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
