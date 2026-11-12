from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from lucidbench.annotation_schema import (
    paired_agreement,
    validate_annotation_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate annotation CSV and compute agreement only from supplied ratings."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--agreement-output")
    args = parser.parse_args()

    path = Path(args.annotations)
    annotations = pd.read_csv(path, dtype=str, keep_default_na=False)
    errors = validate_annotation_frame(
        annotations, require_complete=args.require_complete
    )
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        sys.exit(1)

    annotator_ids = sorted(annotations["annotator_id"].unique())
    result = {
        "valid": True,
        "n_rows": len(annotations),
        "annotator_ids": annotator_ids,
    }
    if len(annotator_ids) == 2:
        result["agreement"] = paired_agreement(annotations, annotator_ids=annotator_ids)
        has_paired_scores = any(
            details["n_paired"] > 0
            for details in result["agreement"]["fields"].values()
        )
        result["agreement_status"] = (
            "computed from supplied annotations"
            if has_paired_scores
            else "not computed: no completed paired annotation scores"
        )
    else:
        result["agreement"] = None
        result["agreement_status"] = (
            "not computed: agreement requires exactly two annotators"
        )

    if args.agreement_output:
        output = Path(args.agreement_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
