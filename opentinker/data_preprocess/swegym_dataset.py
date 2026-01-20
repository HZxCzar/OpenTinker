#!/usr/bin/env python3
"""Prepare SWE-Gym dataset into OpenTinker parquet format."""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

import datasets


DEFAULT_SYSTEM_PROMPT = (
    "You are a software engineer. Solve the issue by modifying the repository."
)


def _load_instances(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        instances = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instances.append(json.loads(line))
        return instances

    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["instances", "data", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported SWE-Gym JSON format.")


def _format_prompt(instance: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    problem_statement = instance.get("problem_statement", "")

    user_prompt = (
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n\n"
        f"Problem statement:\n{problem_statement}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_rows(
    instances: Iterable[Dict[str, Any]], system_prompt: str
) -> List[Dict[str, Any]]:
    rows = []
    for instance in instances:
        rows.append(
            {
                "prompt": _format_prompt(instance, system_prompt),
                "instance": instance,
                "data_source": "swegym",
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional system prompt for SWE-Gym prompts.",
    )

    args = parser.parse_args()

    instances = _load_instances(args.input_path)
    rows = _build_rows(instances, args.system_prompt)

    dataset = datasets.Dataset.from_list(rows)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    dataset.to_parquet(args.output_path)

    print(f"Wrote {len(dataset)} samples to {args.output_path}")
