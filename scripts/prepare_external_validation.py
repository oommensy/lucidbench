from __future__ import annotations

import argparse
import json

from lucidbench.external_validation import prepare_external_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a ratings-free external-validation packet and private linkage manifest."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/external_validation.example.yaml",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    summary = prepare_external_validation(
        args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
