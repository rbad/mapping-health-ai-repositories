#!/usr/bin/env python3
"""
GitHub Health AI Repository Ingestion

- Searches GitHub for health-AI-related repositories using configurable health and ML keyword lists.
- Applies automated filters for activity and size.
- Fetches contributors and README text for each repo.
- Writes a JSONL file with rich metadata, suitable for later graph construction.

Usage example:
    python ingest_health_ai_repos.py --subdomain imaging --output repos_health_ai_imaging.jsonl

The subdomain argument is for tagging all records in an ingestion; can be used for later cross-subdomain analysis.

IMPORTANT: Adjust the HEALTH_TERMS list in the code to match the desired health subdomain (e.g., imaging, ehr, genetics, general).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

# ===========================
# CONFIGURABLE KEYWORD LISTS
# ===========================

# IMPORTANT: Modify this list BEFORE EACH RUN to focus on a specific health subdomain.
# Examples
#   HEALTH_TERMS = ["medical imaging", "radiology", "radiological", "xray", "x-ray", "computed tomography", "ct scan", "mri", "ultrasound", "digital pathology"] # imaging
#   HEALTH_TERMS = ["electronic health record", "EHR", "electronic medical record", "EMR", "clinical documentation", "clinical notes", "clinical text", "discharge summary", "ICD coding", "FHIR", "HL7", "medical coding", "medical billing", "claims data"] # ehr
#   HEALTH_TERMS = ["genomics", "genetic sequencing", "DNA sequencing", "bioinformatics", "transcriptomics", "RNA sequencing", "variant calling", "GWAS", "PCR", "single cell", "proteomics", "CRISPR", "nucleic acid transcription"] # genetics
#   HEALTH_TERMS = ["clinical decision support", "healthcare analytics", "patient monitoring", "medical time series", "vital signs", "hospital readmission", "ICU mortality", "ER mortality", "patient mortality", "sepsis", "disease risk", "triage system", "public health"] # general
HEALTH_TERMS = ["term1", "term2", "term3"]

# Machine learning / AI related terms (keep the same across runs)
ML_TERMS = [
    "machine learning",
    "deep learning",
    "neural network",
    "AI",
    "artificial intelligence"
]

# Languages to prioritize (GitHub "language:" qualifier)
# Adjust if repositories of interest use other languages.
PRIMARY_LANGUAGE: str = ""      # Alternatively, could zoom in on just "Python" projects

# Activity filter
MIN_STARS: int = 10

# Repositories must have been pushed after START_YEAR
START_YEAR: int = 2018
END_YEAR: int = 2025

# Search configuration
MAX_REPOS_PER_QUERY: int = 1000   # max repositories to fetch per (health_term, ml_term) pair (nax 1000)
PER_PAGE: int = 100               # GitHub Search API per_page (max 100)
MAX_PAGES: int = 10               # safety cap on pages per query

# Contributors / README configuration
MAX_CONTRIBUTORS_PER_REPO: int = 50
README_MAX_CHARS: int = 10000    # truncate long READMEs

# GitHub API configuration
GITHUB_API_BASE: str = "https://api.github.com"

# Authentication:
# Preferred: set environment variable GITHUB_TOKEN to a personal access token.
# Alternative: hard-code a token here (not recommended to commit this).
GITHUB_TOKEN: Optional[str] = os.environ.get("GITHUB_TOKEN") or "YOUR_GITHUB_TOKEN_HERE"


# ===========================
# DATA STRUCTURES
# ===========================

@dataclass
class RepoRecord:
    """Single repository record to be saved as JSONL."""
    repo_id: int
    full_name: str
    name: str
    license_spdx_id: Optional[str]
    license_name: Optional[str]    
    description: Optional[str]
    html_url: str
    language: Optional[str]
    topics: List[str]
    stargazers_count: int
    watchers_count: int
    open_issues_count: int
    forks_count: int
    is_fork: bool
    parent_full_name: Optional[str]
    created_at: str
    updated_at: str
    pushed_at: str
    health_subdomain: str
    # Matched terms (for transparency)
    matched_health_terms: List[str]
    matched_ml_terms: List[str]
    # Contributors and README
    contributors: List[str]
    readme_text: Optional[str]


# ===========================
# HELPER FUNCTIONS
# ===========================

def make_github_session() -> requests.Session:
    """Create a configured requests session for GitHub API."""
    if not GITHUB_TOKEN or GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print(
            "WARNING: GitHub token not set or left as placeholder.\n"
            "Set the GITHUB_TOKEN environment variable or edit GITHUB_TOKEN "
            "in this script for authenticated requests (strongly recommended).",
            file=sys.stderr,
        )

    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "health-ai-graph-ingestion-script",
    }
    if GITHUB_TOKEN and GITHUB_TOKEN != "YOUR_GITHUB_TOKEN_HERE":
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    session.headers.update(headers)
    return session


def robust_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_attempts: int = 3,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """
    Wrapper around session.get that retries on network-level errors
    (e.g., connection aborted, remote closed without response).

    Returns:
        - a Response object on success
        - None if all attempts fail
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            return resp
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(
                f"Network error on GET {url} (attempt {attempt}/{max_attempts}): {e}",
                file=sys.stderr,
            )
            if attempt < max_attempts:
                # simple backoff
                time.sleep(5 * attempt)

    print(
        f"Giving up on GET {url} after {max_attempts} failed attempts "
        f"due to network errors: {last_exc}",
        file=sys.stderr,
    )
    return None


def handle_rate_limit(response: requests.Response) -> None:
    """Handle GitHub rate limiting by sleeping until reset, if necessary."""
    if response.status_code != 403:
        return

    remaining = response.headers.get("X-RateLimit-Remaining")
    reset_ts = response.headers.get("X-RateLimit-Reset")

    # Only attempt to sleep if rate limit is clearly exhausted.
    if remaining == "0" and reset_ts is not None:
        reset_time = int(reset_ts)
        now = int(time.time())
        wait_seconds = max(0, reset_time - now) + 5  # small buffer

        # Use timezone-aware datetime in UTC (Python 3.11+ style)
        reset_dt = datetime.fromtimestamp(reset_time, timezone.utc)\
                           .isoformat()\
                           .replace("+00:00", "Z")

        print(
            f"Rate limit exceeded. Reset at {reset_dt}. "
            f"Sleeping for ~{wait_seconds} seconds...",
            file=sys.stderr,
        )
        time.sleep(wait_seconds)


def search_repositories(
    session: requests.Session,
    health_terms: List[str],
    ml_terms: List[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Use the GitHub Search API for combinations of health_terms, ml_terms, and year segments.
    Deduplicate by repository ID across all segments.
    Returns a dict: repo_id -> raw repo JSON.
    """
    repos: Dict[int, Dict[str, Any]] = {}

    for h_term in health_terms:
        for ml_term in ml_terms:
            for year in range(START_YEAR, END_YEAR + 1):
                # GitHub Search API hard cap: only first 1000 results per query.
                max_results_for_segment = min(MAX_REPOS_PER_QUERY, 1000)

                # pushed:YYYY-01-01..YYYY-12-31
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                pushed_clause = f"pushed:{start_date}..{end_date} "

                if PRIMARY_LANGUAGE:
                    language_clause = f"language:{PRIMARY_LANGUAGE} "
                else:
                    language_clause = ""

                query = (
                    f'{h_term} "{ml_term}" '
                    f'in:name,description,readme '
                    f'{language_clause}'
                    f'{pushed_clause}'
                    f'stars:>={MIN_STARS} '
                    f'archived:false'
                )
                print(f"\n[SEARCH] Query: {query}", file=sys.stderr)

                fetched_for_segment = 0

                for page in range(1, MAX_PAGES + 1):
                    # Do not attempt another request if already at cap
                    if fetched_for_segment >= max_results_for_segment:
                        break

                    params = {
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": PER_PAGE,
                        "page": page,
                    }
                    url = f"{GITHUB_API_BASE}/search/repositories"

                    response = robust_get(session, url, params=params)
                    if response is None:
                        # network kept failing; stop this segment and move on
                        break

                    if response.status_code == 403:
                        handle_rate_limit(response)
                        response = robust_get(session, url, params=params)
                        if response is None:
                            print(
                                f"Failed to recover after rate limit for query '{query}' "
                                f"(year={year}); skipping remaining pages for this segment.",
                                file=sys.stderr,
                            )
                            break

                    # Handle the 1000-results limit per query gracefully
                    if response.status_code == 422:
                        # Try to inspect message, but be robust to non-JSON
                        try:
                            msg = response.json().get("message", "")
                        except ValueError:
                            msg = response.text or ""

                        if "Only the first 1000 search results are available" in msg:
                            print(
                                f"Reached GitHub 1000-result limit for this query "
                                f"(year={year}); stopping pagination for this segment.",
                                file=sys.stderr,
                            )
                            break
                        else:
                            print(
                                f"Unprocessable Entity for query '{query}' "
                                f"(year={year}, page={page}): {msg}",
                                file=sys.stderr,
                            )
                            break

                    try:
                        response.raise_for_status()
                    except requests.HTTPError as e:
                        print(
                            f"Error during search (year={year}, page={page}): "
                            f"{e} - {response.text}",
                            file=sys.stderr,
                        )
                        break

                    data = response.json()
                    items = data.get("items", [])
                    if not items:
                        break

                    for repo in items:
                        repo_id = repo["id"]
                        if repo_id not in repos:
                            repos[repo_id] = repo
                        # Count per-segment, even if repo already seen in another segment
                        fetched_for_segment += 1
                        if fetched_for_segment >= max_results_for_segment:
                            break

                    print(
                        f"  Year {year}, page {page}: fetched {len(items)} items; "
                        f"unique repos total = {len(repos)}",
                        file=sys.stderr,
                    )

                    # Stop if fewer results than per_page (end of pages for this segment)
                    if len(items) < PER_PAGE:
                        break

                print(
                    f"[SEARCH] Done with query '{h_term}' + '{ml_term}' for year {year}: "
                    f"{fetched_for_segment} repos (capped at {max_results_for_segment}).",
                    file=sys.stderr,
                )

    print(f"\n[SEARCH] Total unique repositories collected: {len(repos)}", file=sys.stderr)
    return repos


def passes_basic_filters(repo: Dict[str, Any]) -> bool:
    """Apply basic activity and size filters using repo metadata."""
    # Stars filter
    if repo.get("stargazers_count", 0) < MIN_STARS:
        return False

    # Ensure pushed_at exists and is parseable (sanity check)
    pushed_at_str = repo.get("pushed_at")
    if pushed_at_str is None:
        return False

    try:
        # Parse just to validate format; no date cutoff here anymore
        datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
    except ValueError:
        return False

    return True


def fetch_contributors_for_repo(
    session: requests.Session,
    full_name: str,
) -> List[str]:
    """Fetch up to MAX_CONTRIBUTORS_PER_REPO contributor logins for a repository."""
    contributors: List[str] = []
    page = 1

    while len(contributors) < MAX_CONTRIBUTORS_PER_REPO:
        url = f"{GITHUB_API_BASE}/repos/{full_name}/contributors"
        params = {"per_page": 100, "page": page}

        response = robust_get(session, url, params=params)
        if response is None:
            print(
                f"Giving up on contributors for {full_name} due to repeated network errors.",
                file=sys.stderr,
            )
            break

        if response.status_code == 403:
            handle_rate_limit(response)
            response = robust_get(session, url, params=params)
            if response is None:
                print(
                    f"Failed to recover after rate limit while fetching contributors for "
                    f"{full_name}.",
                    file=sys.stderr,
                )
                break

        # If the repo or contributor list is not accessible, just stop trying
        if response.status_code in (404, 451):
            break

        # 204 No Content or any other 2xx with empty body -> nothing useful
        if response.status_code == 204:
            break

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(
                f"Error fetching contributors for {full_name}: {e} "
                f"(status={response.status_code})",
                file=sys.stderr,
            )
            break

        # Now safely attempt to parse JSON
        try:
            data = response.json()
        except ValueError:
            # Non-JSON or empty response body; log and stop for this repo
            text_preview = (response.text or "").strip()[:200]
            print(
                f"Warning: non-JSON response from contributors endpoint for {full_name} "
                f"(status={response.status_code}). Response preview: {text_preview!r}",
                file=sys.stderr,
            )
            break

        if not isinstance(data, list) or not data:
            # No contributors on this page; we're done
            break

        for contributor in data:
            login = contributor.get("login")
            if login:
                contributors.append(login)
                if len(contributors) >= MAX_CONTRIBUTORS_PER_REPO:
                    break

        if len(data) < 100:
            # Fewer than a full page ⇒ no more pages
            break

        page += 1

    return contributors


def fetch_readme_for_repo(session: requests.Session, full_name: str) -> Optional[str]:
    """Fetch README text for a repository (truncated)."""
    url = f"{GITHUB_API_BASE}/repos/{full_name}/readme"
    # Ask for raw content instead of base64 JSON
    headers = {"Accept": "application/vnd.github.v3.raw"}

    response = robust_get(session, url, headers=headers)
    if response is None:
        print(
            f"Giving up on README for {full_name} due to repeated network errors.",
            file=sys.stderr,
        )
        return None

    if response.status_code == 403:
        handle_rate_limit(response)
        response = robust_get(session, url, headers=headers)
        if response is None:
            print(
                f"Failed to recover after rate limit while fetching README for "
                f"{full_name}.",
                file=sys.stderr,
            )
            return None

    if response.status_code == 404:
        return None

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"Error fetching README for {full_name}: {e}", file=sys.stderr)
        return None

    text = response.text
    if text is None:
        return None
    if len(text) > README_MAX_CHARS:
        text = text[:README_MAX_CHARS]
    return text


def match_terms(text: str, terms: List[str]) -> List[str]:
    """Return the subset of terms that appear in text (case-insensitive substring match)."""
    text_lower = text.lower()
    matched = [term for term in terms if term.lower() in text_lower]
    return matched


def final_health_ai_filter(
    repo: Dict[str, Any],
    readme_text: Optional[str],
    health_terms: List[str],
    ml_terms: List[str],
) -> (bool, List[str], List[str]):
    """
    Final automated filters to ensure repository is truly health + AI related.
    Returns:
        (keep_repo, matched_health_terms, matched_ml_terms)
    """
    name = repo.get("name") or ""
    desc = repo.get("description") or ""
    combined_text = f"{name}\n{desc}\n{readme_text or ''}"

    matched_health = match_terms(combined_text, health_terms)
    matched_ml = match_terms(combined_text, ml_terms)

    # Require at least one match from each list
    keep = bool(matched_health) and bool(matched_ml)
    return keep, matched_health, matched_ml


def extract_repo_record(
    repo: Dict[str, Any],
    subdomain: str,
    matched_health_terms: List[str],
    matched_ml_terms: List[str],
    contributors: List[str],
    readme_text: Optional[str],
) -> RepoRecord:
    """Convert raw repo JSON to RepoRecord dataclass."""
    # Topics might be under repo["topics"], but for some endpoints need preview header.
    topics = repo.get("topics", [])
    if not isinstance(topics, list):
        topics = []

    parent_full_name = None
    if repo.get("fork") and isinstance(repo.get("parent"), dict):
        parent_full_name = repo["parent"].get("full_name")

    license_info = repo.get("license") or {}
    license_spdx = license_info.get("spdx_id")
    license_name = license_info.get("name")    

    return RepoRecord(
        repo_id=repo["id"],
        full_name=repo["full_name"],
        name=repo.get("name", ""),
        license_spdx_id=license_spdx,
        license_name=license_name,        
        description=repo.get("description"),
        html_url=repo.get("html_url", ""),
        language=repo.get("language"),
        topics=topics,
        stargazers_count=repo.get("stargazers_count", 0),
        watchers_count=repo.get("watchers_count", 0),
        open_issues_count=repo.get("open_issues_count", 0),
        forks_count=repo.get("forks_count", 0),
        is_fork=bool(repo.get("fork", False)),
        parent_full_name=parent_full_name,
        created_at=repo.get("created_at", ""),
        updated_at=repo.get("updated_at", ""),
        pushed_at=repo.get("pushed_at", ""),
        health_subdomain=subdomain,
        matched_health_terms=matched_health_terms,
        matched_ml_terms=matched_ml_terms,
        contributors=contributors,
        readme_text=readme_text,
    )


# ===========================
# MAIN EXECUTION
# ===========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GitHub repositories for a health AI subdomain."
    )
    parser.add_argument(
        "--subdomain",
        required=True,
        help="Label for this run (e.g., imaging, ehr, genetics, general).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file (one repository per line).",
    )
    args = parser.parse_args()

    session = make_github_session()

    # Search and deduplicate
    raw_repos = search_repositories(session, HEALTH_TERMS, ML_TERMS)

    # Filter, fetch contributors and README, final filter, and save
    output_path = args.output
    kept_count = 0
    total_considered = 0

    # Track how often each health term matches kept repos
    health_term_counts = defaultdict(int)
    # Track how many kept repos match only exactly one health term
    health_term_exclusive_counts = defaultdict(int)    

    with open(output_path, "w", encoding="utf-8") as f_out:
        for repo_id, repo in raw_repos.items():
            total_considered += 1

            if not passes_basic_filters(repo):
                continue

            full_name = repo.get("full_name")
            if not full_name:
                continue

            # contributors
            contributors = fetch_contributors_for_repo(session, full_name)

            # README
            readme_text = fetch_readme_for_repo(session, full_name)

            # final health + AI filter
            keep, matched_health, matched_ml = final_health_ai_filter(
                repo,
                readme_text,
                HEALTH_TERMS,
                ML_TERMS,
            )
            if not keep:
                continue

            # Update health-term statistics for kept repos
            # matched_health is a list of terms that were found in name/description/README
            health_terms_set = set(matched_health)
            for term in health_terms_set:
                health_term_counts[term] += 1
            if len(health_terms_set) == 1:
                # Repo matched exactly one health term
                only_term = next(iter(health_terms_set))
                health_term_exclusive_counts[only_term] += 1             

            record = extract_repo_record(
                repo=repo,
                subdomain=args.subdomain,
                matched_health_terms=matched_health,
                matched_ml_terms=matched_ml,
                contributors=contributors,
                readme_text=readme_text,
            )

            json_line = json.dumps(asdict(record), ensure_ascii=False)
            f_out.write(json_line + "\n")
            kept_count += 1

            print(
                f"[KEPT] [#{kept_count}] {record.full_name} (stars={record.stargazers_count}, "
                f"contributors={len(contributors)})",
                file=sys.stderr,
            )

    print(
        f"\nDone. Considered {total_considered} repos; kept {kept_count} after all filters.\n"
        f"Output written to: {output_path}",
        file=sys.stderr,
    )

    # Report health-term statistics for kept repos
    if health_term_counts:
        print("\n[STATS] Health term matches among KEPT repositories:", file=sys.stderr)
        for term in HEALTH_TERMS:
            total = health_term_counts.get(term, 0)
            exclusive = health_term_exclusive_counts.get(term, 0)
            print(
                f"  - '{term}': matched in {total} kept repos "
                f"(exclusive: {exclusive})",
                file=sys.stderr,
            )

        # Optional: also dump to a JSON file next to the output
        stats_output = output_path + ".health_term_stats.json"
        stats_payload = {
            "health_term_counts": dict(health_term_counts),
            "health_term_exclusive_counts": dict(health_term_exclusive_counts),
        }
        with open(stats_output, "w", encoding="utf-8") as f_stats:
            json.dump(stats_payload, f_stats, indent=2, ensure_ascii=False)
        print(
            f"[STATS] Detailed health-term stats written to {stats_output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
