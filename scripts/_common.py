"""Shared entry-point helpers of the scripts."""

from __future__ import annotations

import json

import pandas as pd


def show(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
            print(pd.DataFrame(payload).to_string(index=False))
    else:
        print(json.dumps(payload, indent=2, default=str))
