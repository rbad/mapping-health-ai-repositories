#!/usr/bin/env python3
"""
Merge multiple ingested health-AI subdomain JSONL files into a single deduplicated dataset.

- Loads all input JSONL files into memory (one dict per repo).
- Deduplicates by `repo_id` (falls back to `full_name` if `repo_id` is missing).
- For repos that appear in multiple subdomain files, aggregates all subdomains into a `health_subdomains` list.
- For repos that only appear once, the value in health_subdomain is simply copied to 'health_subdomains' (i.e. a one -element list).

Example usage:

    python merge_ingested_health_ai_repos.py \
        --inputs repos_health_ai_imaging.jsonl \
                 repos_health_ai_ehr.jsonl \
                 repos_health_ai_genetics.jsonl \
                 repos_health_ai_general.jsonl \
        --output repos_health_ai_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def get_subdomains_from_record(rec: Dict[str, Any]) -> List[str]:
    """
    Extract subdomain labels from a record.

    - Uses `health_subdomain` if present.
    - Also merges any existing `health_subdomains` list (for robustness).
    - Returns a list of unique subdomains (order not guaranteed).
    """
    subs = []

    single = rec.get("health_subdomain")
    if isinstance(single, str) and single.strip():
        subs.append(single.strip())

    multi = rec.get("health_subdomains")
    if isinstance(multi, list):
        for x in multi:
            if isinstance(x, str) and x.strip():
                subs.append(x.strip())

    # Deduplicate while preserving order of first appearance
    seen = set()
    unique_subs: List[str] = []
    for s in subs:
        if s not in seen:
            seen.add(s)
            unique_subs.append(s)
    return unique_subs


def merge_records(existing: Dict[str, Any], new: Dict[str, Any]) -> None:
    """
    Merge `new` record into `existing` in-place by:
    - Aggregating `health_subdomains`.
    - Leaving other fields as they are in `existing` (first-seen record wins).

    If you later want a more sophisticated merge (e.g., prefer non-null values,
    reconcile conflicting fields), you can extend this function.
    """
    existing_subs = set(existing.get("health_subdomains") or [])
    new_subs = get_subdomains_from_record(new)
    for s in new_subs:
        existing_subs.add(s)

    existing["health_subdomains"] = sorted(existing_subs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple health-AI subdomain JSONL files into one deduplicated file."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files (e.g., imaging, ehr, genetics, general).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file with merged repositories.",
    )
    args = parser.parse_args()

    # Primary index: repo_id -> record
    repos_by_id: Dict[int, Dict[str, Any]] = {}
    # Fallback index for records without repo_id: full_name -> record
    repos_by_full_name: Dict[str, Dict[str, Any]] = {}

    total_records = 0
    duplicate_by_id = 0
    fallback_used = 0
    duplicate_by_full_name = 0

    for path in args.inputs:
        print(f"[LOAD] Reading from {path} ...", file=sys.stderr)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_records += 1

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: skipping invalid JSON line in {path}: {e}",
                        file=sys.stderr,
                    )
                    continue

                repo_id = rec.get("repo_id")
                full_name = rec.get("full_name")

                if repo_id is not None:
                    # Primary deduplication key
                    if repo_id in repos_by_id:
                        # Merge subdomains into existing record
                        duplicate_by_id += 1
                        merge_records(repos_by_id[repo_id], rec)
                    else:
                        # New repo: initialize health_subdomains
                        merged_rec = dict(rec)  # shallow copy
                        merged_rec["health_subdomains"] = get_subdomains_from_record(rec)
                        repos_by_id[repo_id] = merged_rec
                else:
                    # Fallback path: no repo_id, deduplicate by full_name
                    if not full_name:
                        # No usable key; just skip this weird record
                        print(
                            "Warning: record without repo_id and full_name encountered; skipping.",
                            file=sys.stderr,
                        )
                        continue

                    fallback_used += 1
                    if full_name in repos_by_full_name:
                        duplicate_by_full_name += 1
                        merge_records(repos_by_full_name[full_name], rec)
                    else:
                        merged_rec = dict(rec)
                        merged_rec["health_subdomains"] = get_subdomains_from_record(rec)
                        repos_by_full_name[full_name] = merged_rec

    # Combine primary and fallback maps into one list of records
    merged_records: List[Dict[str, Any]] = []
    merged_records.extend(repos_by_id.values())
    merged_records.extend(repos_by_full_name.values())

    print(
        f"[SUMMARY] Total input records: {total_records}",
        file=sys.stderr,
    )
    print(
        f"[SUMMARY] Unique repos by repo_id: {len(repos_by_id)} "
        f"(duplicates by id: {duplicate_by_id})",
        file=sys.stderr,
    )
    if fallback_used > 0:
        print(
            f"[SUMMARY] Records without repo_id (fallback full_name): {fallback_used}, "
            f"unique fallback repos: {len(repos_by_full_name)}, "
            f"duplicates by full_name: {duplicate_by_full_name}",
            file=sys.stderr,
        )

    print(f"[WRITE] Writing merged output to {args.output} ...", file=sys.stderr)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for rec in merged_records:
            # Ensure health_subdomains is present and consistent for all records
            subs = get_subdomains_from_record(rec)
            rec["health_subdomains"] = subs
            json_line = json.dumps(rec, ensure_ascii=False)
            f_out.write(json_line + "\n")

    print(
        f"[DONE] Merged {len(merged_records)} unique repositories into {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
