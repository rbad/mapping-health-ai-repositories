#!/usr/bin/env python3
"""
Cross-subdomain analysis for health-AI GitHub repositories.

Given a merged JSONL file (e.g., repos_health_ai_merged.jsonl) where each line represents a repository record
(as produced by ingest_health_ai_repos.py then merge_ingested_health_ai_repos.py), this script:

1. Builds a global repository-level collaboration network using shared contributors.
2. Derives per-repo primary_subdomain from listed health_subdomains.
3. Computes:
   - Subdomain-subdomain connectivity matrix (edge counts and total weights).
   - Per-repo cross-subdomain neighbor stats.
   - Louvain communities on the largest connected component and their subdomain composition.
4. Generates a global graph:
   - Nodes = ~top 300 most structurally important repos,
   - Node size = importance (degree + cross-subdomain neighbors),
   - Edges = contributor-based collaborations,
   - Layout = force-directed, so communities naturally appear as tight clusters,
   - Labels = names of the top ~15 hubs.

Outputs:
    - cross_summary.json
    - subdomain_connectivity_matrix.csv
    - cross_subdomain_node_stats.csv
    - cross_communities_subdomain_mix.csv
    - cross_community_node_labels.csv
    - subdomain_connectivity_matrix.png
    - core_network_graph.png

Usage example:
    python analyze_cross_health_ai_subdomains.py --input repos_health_ai_merged.jsonl --output-dir cross_subdomain_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_comm


# ---------- Subdomain key terms for primary_subdomain inference ----------

SUBDOMAIN_KEY_TERMS = {
    "imaging": ["medical imaging", "radiology", "radiological", "xray", "x-ray", "computed tomography", "ct scan", "mri", "ultrasound", "digital pathology"],
    "ehr": ["electronic health record", "EHR", "electronic medical record", "EMR", "clinical documentation", "clinical notes", "clinical text", "discharge summary", "ICD coding", "FHIR", "HL7", "medical coding", "medical billing", "claims data"],
    "genetics": ["genomics", "genetic sequencing", "DNA sequencing", "bioinformatics", "transcriptomics", "RNA sequencing", "variant calling", "GWAS", "PCR", "single cell", "proteomics", "CRISPR", "nucleic acid transcription"],
    "general": ["clinical decision support", "healthcare analytics", "patient monitoring", "medical time series", "vital signs", "hospital readmission", "ICU mortality", "ER mortality", "patient mortality", "sepsis", "disease risk", "triage system", "public health"]
}


# ---------- Data loading & helpers ----------

def load_repos(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load repositories from a JSONL file into a list of dicts."""
    repos: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                repos.append(rec)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping invalid JSON line in {jsonl_path}: {e}",
                    file=sys.stderr,
                )
    return repos


def get_health_subdomains(rec: Dict[str, Any]) -> List[str]:
    """
    Extract subdomain labels from a record.

    Uses:
      - rec["health_subdomain"] if present and non-empty,
      - rec["health_subdomains"] list if present.
    Returns a deduplicated list of subdomains.
    """
    subs: List[str] = []

    single = rec.get("health_subdomain")
    if isinstance(single, str) and single.strip():
        subs.append(single.strip())

    multi = rec.get("health_subdomains")
    if isinstance(multi, list):
        for s in multi:
            if isinstance(s, str) and s.strip():
                subs.append(s.strip())

    seen = set()
    unique_subs: List[str] = []
    for s in subs:
        if s not in seen:
            seen.add(s)
            unique_subs.append(s)
    return unique_subs


def determine_primary_subdomain(rec: Dict[str, Any]) -> str:
    """
    Determine a single 'primary_subdomain' label for a repository.

    Strategy:
      1. Get the deduplicated list of health_subdomains for the record.
      2. If exactly one subdomain, return it.
      3. If multiple, score each candidate subdomain by counting occurrences
         of its key terms in the repository's description + README text.
         The subdomain with the highest score is chosen.
      4. If all scores are zero or tied, prefer more specific subdomains over
         'general', then fall back to a fixed priority order.
    """
    subs = get_health_subdomains(rec)
    if not subs:
        return "unknown"
    if len(subs) == 1:
        return subs[0]

    # Combined text from description + README
    description = rec.get("description") or ""
    readme = rec.get("readme_text") or ""
    text = f"{description}\n{readme}".lower()

    best_sub = None
    best_score = -1

    for s in subs:
        key_terms = SUBDOMAIN_KEY_TERMS.get(s, [])
        score = 0
        for term in key_terms:
            term_l = term.lower()
            if not term_l:
                continue
            # simple substring count
            score += text.count(term_l)
        if score > best_score:
            best_score = score
            best_sub = s

    # If we got a positive-scoring winner, use it
    if best_sub is not None and best_score > 0:
        return best_sub

    # Otherwise, fall back to a rule-based choice:
    # 1) prefer subdomains other than 'general' if available
    non_general = [s for s in subs if s != "general"]
    candidates = non_general or subs

    # 2) fixed priority order
    priority_order = ["imaging", "ehr", "genetics", "general", "unknown"]
    for p in priority_order:
        if p in candidates:
            return p

    # 3) final fallback: first in the list
    return candidates[0]


def is_bot_contributor(login: str) -> bool:
    """Heuristic to ignore obvious bots / automation accounts."""
    if not login:
        return False
    name = login.lower()
    if "[bot]" in name:
        return True
    if name.endswith("bot"):
        return True
    known = {
        "github-actions[bot]",
        "dependabot[bot]",
        "actions-user",
        "gitter-badger",
        "readmecritic",
    }
    if name in known:
        return True
    return False


# ---------- Graph construction ----------

def build_global_repo_graph(
    repos: Sequence[Dict[str, Any]],
    min_shared_contributors: int = 1,
    include_parent_fork_edges: bool = True,
) -> Tuple[nx.Graph, Dict[int, Dict[str, Any]]]:
    """
    Build an undirected graph for all subdomains combined.

    Nodes: repositories (repo_id), with attributes including:
        - full_name
        - health_subdomains
        - primary_subdomain
        - stargazers_count, etc.

    Edges:
        - Weighted by the number of shared contributors between two repos.
        - parent_fork indicator based on parent_full_name when available.
    """
    G = nx.Graph()
    id_to_repo: Dict[int, Dict[str, Any]] = {}
    full_name_to_id: Dict[str, int] = {}

    # Add nodes
    for rec in repos:
        rid = rec.get("repo_id")
        if rid is None:
            # skip weird records that somehow have no repo_id
            continue
        rid = int(rid)

        subs = get_health_subdomains(rec)
        primary = determine_primary_subdomain(rec)

        id_to_repo[rid] = rec
        full_name = rec.get("full_name", "")
        if isinstance(full_name, str) and full_name:
            full_name_to_id[full_name] = rid

        G.add_node(
            rid,
            full_name=full_name,
            name=rec.get("name"),
            health_subdomains=subs,
            primary_subdomain=primary,
            description=rec.get("description"),
            language=rec.get("language"),
            topics=",".join(rec.get("topics") or []),
            stargazers_count=rec.get("stargazers_count", 0),
            watchers_count=rec.get("watchers_count", 0),
            open_issues_count=rec.get("open_issues_count", 0),
            forks_count=rec.get("forks_count", 0),
            license_spdx_id=rec.get("license_spdx_id"),
            license_name=rec.get("license_name"),
        )

    # Map contributors to repos
    contrib_to_repos: Dict[str, Set[int]] = defaultdict(set)
    for rec in repos:
        rid = rec.get("repo_id")
        if rid is None:
            continue
        rid = int(rid)
        if not G.has_node(rid):
            continue
        for login in rec.get("contributors", []):
            if is_bot_contributor(login):
                continue
            contrib_to_repos[login].add(rid)

    # Build edges from shared contributors
    edge_weights: Dict[Tuple[int, int], int] = defaultdict(int)
    for login, repo_ids in contrib_to_repos.items():
        ids = sorted(repo_ids)
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a == b:
                    continue
                edge_weights[(a, b)] += 1

    for (a, b), w in edge_weights.items():
        if w < min_shared_contributors:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
            G[a][b]["shared_contributors"] += w
        else:
            G.add_edge(
                a,
                b,
                weight=w,
                shared_contributors=w,
                parent_fork=False,
            )

    # Optional: parent-fork edges
    if include_parent_fork_edges:
        for rec in repos:
            rid = rec.get("repo_id")
            if rid is None:
                continue
            rid = int(rid)
            if not G.has_node(rid):
                continue
            parent_full_name = rec.get("parent_full_name")
            if not parent_full_name:
                continue
            parent_id = full_name_to_id.get(parent_full_name)
            if parent_id is None or parent_id == rid:
                continue
            if G.has_edge(rid, parent_id):
                G[rid][parent_id]["weight"] += 1
                G[rid][parent_id]["parent_fork"] = True
            else:
                G.add_edge(
                    rid,
                    parent_id,
                    weight=1,
                    shared_contributors=0,
                    parent_fork=True,
                )

    return G, id_to_repo


# ---------- Cross-subdomain metrics ----------

def compute_subdomain_connectivity_matrix(G: nx.Graph) -> pd.DataFrame:
    """
    For each edge, accumulate statistics between primary_subdomains.

    Returns a DataFrame with:
        subdomain_u, subdomain_v, edge_count, total_weight
    where (subdomain_u, subdomain_v) is an unordered pair (u <= v lexicographically).
    """
    pair_edge_count: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_weight_sum: Dict[Tuple[str, str], float] = defaultdict(float)

    for u, v, data in G.edges(data=True):
        su = G.nodes[u].get("primary_subdomain", "unknown")
        sv = G.nodes[v].get("primary_subdomain", "unknown")
        pair = tuple(sorted((su, sv)))
        w = float(data.get("weight", 1.0))
        pair_edge_count[pair] += 1
        pair_weight_sum[pair] += w

    records: List[Dict[str, Any]] = []
    for (su, sv), cnt in pair_edge_count.items():
        records.append(
            {
                "subdomain_u": su,
                "subdomain_v": sv,
                "edge_count": cnt,
                "total_weight": pair_weight_sum[(su, sv)],
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def compute_node_cross_subdomain_stats(G: nx.Graph) -> pd.DataFrame:
    """
    For each node (repository), compute:
        - primary_subdomain
        - health_subdomains (joined)
        - intrinsic_subdomain_count (len(health_subdomains))
        - is_intrinsic_multi_subdomain (bool)
        - total_neighbors
        - cross_subdomain_neighbors (neighbors with different primary_subdomain)
        - cross_subdomain_ratio
        - cross_weighted_degree (sum of weights to neighbors of a different primary_subdomain)
    """
    records: List[Dict[str, Any]] = []

    for nid in G.nodes():
        attrs = G.nodes[nid]
        primary = attrs.get("primary_subdomain", "unknown")
        health_subdomains = attrs.get("health_subdomains") or []
        if not isinstance(health_subdomains, list):
            health_subdomains = []
        health_subdomains = [str(s) for s in health_subdomains]
        intrinsic_count = len(health_subdomains)
        is_intrinsic_multi = intrinsic_count > 1

        neighbors = list(G.neighbors(nid))
        total_neighbors = len(neighbors)
        cross_neighbors = 0
        cross_weighted_degree = 0.0

        for nbr in neighbors:
            nbr_primary = G.nodes[nbr].get("primary_subdomain", "unknown")
            w = float(G[nid][nbr].get("weight", 1.0))
            if nbr_primary != primary:
                cross_neighbors += 1
                cross_weighted_degree += w

        cross_ratio = (
            float(cross_neighbors) / float(total_neighbors)
            if total_neighbors > 0
            else 0.0
        )

        records.append(
            {
                "repo_id": int(nid),
                "full_name": attrs.get("full_name"),
                "primary_subdomain": primary,
                "health_subdomains": ", ".join(health_subdomains),
                "intrinsic_subdomain_count": intrinsic_count,
                "is_intrinsic_multi_subdomain": bool(is_intrinsic_multi),
                "total_neighbors": total_neighbors,
                "cross_subdomain_neighbors": cross_neighbors,
                "cross_subdomain_ratio": cross_ratio,
                "cross_weighted_degree": cross_weighted_degree,
                "stargazers_count": attrs.get("stargazers_count", 0),
            }
        )

    return pd.DataFrame.from_records(records)


# ---------- Community detection & mix ----------

def detect_global_communities_and_subdomain_mix(
    G: nx.Graph,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Louvain community detection on the largest connected component
    and summarize subdomain composition of each community.

    Returns:
        communities_mix_df: rows = community_id, with:
            - size
            - primary_subdomain_counts (JSON string)
        node_labels_df: rows = repo_id, community_id
    """
    components = list(nx.connected_components(G))
    if not components:
        return pd.DataFrame(), pd.DataFrame()

    components_sorted = sorted(components, key=len, reverse=True)
    largest_comp_nodes: Set[int] = set(components_sorted[0])
    G_largest = G.subgraph(largest_comp_nodes).copy()

    if G_largest.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    print(
        f"[COMMUNITIES] Running Louvain on largest component with "
        f"{G_largest.number_of_nodes()} nodes and {G_largest.number_of_edges()} edges.",
        file=sys.stderr,
    )
    communities = nx_comm.louvain_communities(
        G_largest,
        weight="weight",
        seed=42,
    )

    community_records: List[Dict[str, Any]] = []
    node_label_records: List[Dict[str, Any]] = []

    for cid, comm_nodes in enumerate(communities):
        subdomain_counts: Dict[str, int] = defaultdict(int)
        for nid in comm_nodes:
            primary = G_largest.nodes[nid].get("primary_subdomain", "unknown")
            subdomain_counts[primary] += 1
            node_label_records.append({"repo_id": int(nid), "community_id": cid})

        community_records.append(
            {
                "community_id": cid,
                "size": len(comm_nodes),
                "subdomain_counts_json": json.dumps(subdomain_counts),
            }
        )

    communities_mix_df = pd.DataFrame.from_records(community_records)
    node_labels_df = pd.DataFrame.from_records(node_label_records)

    return communities_mix_df, node_labels_df


# ---------- Plotting ----------

def plot_subdomain_matrix(
    matrix_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot a heatmap of edge_count between primary_subdomains.

    The matrix is symmetrized so both (A,B) and (B,A) cells are filled.
    """
    if matrix_df.empty:
        print("[PLOT] subdomain connectivity matrix is empty; skipping plot.", file=sys.stderr)
        return

    # Collect all subdomains
    subs: Set[str] = set()
    for _, row in matrix_df.iterrows():
        subs.add(str(row["subdomain_u"]))
        subs.add(str(row["subdomain_v"]))
    subs_sorted = sorted(subs)

    index_map = {s: i for i, s in enumerate(subs_sorted)}
    n = len(subs_sorted)
    mat = np.zeros((n, n), dtype=float)

    # Fill symmetric matrix
    for _, row in matrix_df.iterrows():
        su = str(row["subdomain_u"])
        sv = str(row["subdomain_v"])
        cnt = float(row["edge_count"])
        i = index_map[su]
        j = index_map[sv]
        mat[i, j] += cnt
        mat[j, i] += cnt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(subs_sorted, rotation=45, ha="right")
    ax.set_yticklabels(subs_sorted)
    ax.set_xlabel("Primary subdomain")
    ax.set_ylabel("Primary subdomain")
    ax.set_title("Subdomain–subdomain edge counts")

    # Annotate cells with counts (rounded)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if val > 0:
                text = f"{int(val):d}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_core_network_graph(
    G: nx.Graph,
    node_stats_df: pd.DataFrame,
    node_labels_df: pd.DataFrame,
    output_path: str,
    max_nodes: int = 300,
) -> None:
    """
    Plot a 'core' collaboration network:

    - Nodes: top `max_nodes` repositories by an importance score
             (here: total_neighbors + cross_subdomain_neighbors).
    - Edges: between those selected nodes.
    - Node size: proportional to importance.
    - Labels: a small number of the most important nodes.

    This avoids hairball plots by focusing on the structurally important core.
    """
    if node_stats_df.empty or node_labels_df.empty:
        print("[PLOT] Node stats or node community labels empty; skipping core graph.", file=sys.stderr)
        return

    # Merge community labels in case you want to use them later (right now for info only)
    df = node_stats_df.merge(node_labels_df, on="repo_id", how="inner")
    # Require at least one neighbor
    df = df[df["total_neighbors"] > 0]
    if df.empty:
        print("[PLOT] No nodes with neighbors; skipping core graph.", file=sys.stderr)
        return

    # Define a simple importance metric
    df["importance"] = df["total_neighbors"] + df["cross_subdomain_neighbors"]

    # Select top-N important nodes
    df_top = df.sort_values("importance", ascending=False).head(max_nodes)
    selected_ids = set(df_top["repo_id"].tolist())

    G_sub = G.subgraph(selected_ids).copy()
    if G_sub.number_of_nodes() == 0:
        print("[PLOT] Core subgraph is empty; skipping core graph.", file=sys.stderr)
        return

    # Layout on the subgraph
    print(
        f"[PLOT] Drawing core network graph with {G_sub.number_of_nodes()} nodes "
        f"and {G_sub.number_of_edges()} edges.",
        file=sys.stderr,
    )
    pos = nx.spring_layout(G_sub, weight="weight", seed=42)

    # Node sizes scaled by importance
    imp_series = df_top.set_index("repo_id")["importance"]
    min_imp = float(imp_series.min())
    max_imp = float(imp_series.max())

    def scale_size(imp: float) -> float:
        if max_imp <= min_imp:
            return 300.0
        # Map importance into [100, 1000]
        return 100.0 + 900.0 * (imp - min_imp) / (max_imp - min_imp)

    node_sizes: List[float] = []
    for nid in G_sub.nodes():
        imp = float(imp_series.get(nid, min_imp))
        node_sizes.append(scale_size(imp))

    # Draw
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G_sub, pos, alpha=0.1)
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes)
    # Label a small number of most important nodes
    top_label_ids = set(df_top.head(15)["repo_id"].tolist())
    labels = {
        nid: (G.nodes[nid].get("name") or G.nodes[nid].get("full_name"))
        for nid in G_sub.nodes()
        if nid in top_label_ids
    }
    if labels:
        nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=6)

    plt.axis("off")
    plt.title("Core cross-subdomain repository collaboration network")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------- Main CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-subdomain analysis for merged health-AI GitHub repos."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to merged JSONL file (e.g., repos_health_ai_merged.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where cross-subdomain analysis outputs will be written.",
    )
    parser.add_argument(
        "--min-shared-contributors",
        type=int,
        default=1,
        help="Minimum shared contributors required to create an edge.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[LOAD] Reading merged repositories from {args.input} ...", file=sys.stderr)
    repos = load_repos(args.input)
    print(f"[LOAD] Loaded {len(repos)} repository records.", file=sys.stderr)

    print("[GRAPH] Building global collaboration network ...", file=sys.stderr)
    G, id_to_repo = build_global_repo_graph(
        repos,
        min_shared_contributors=args.min_shared_contributors,
        include_parent_fork_edges=True,
    )
    print(
        f"[GRAPH] Global graph has {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges.",
        file=sys.stderr,
    )

    # Summary of nodes per subdomain
    subdomain_node_counts: Dict[str, int] = defaultdict(int)
    for nid, attrs in G.nodes(data=True):
        primary = attrs.get("primary_subdomain", "unknown")
        subdomain_node_counts[primary] += 1

    # Subdomain connectivity matrix (edges)
    print("[CROSS] Computing subdomain connectivity matrix ...", file=sys.stderr)
    matrix_df = compute_subdomain_connectivity_matrix(G)
    matrix_csv_path = os.path.join(args.output_dir, "subdomain_connectivity_matrix.csv")
    matrix_df.to_csv(matrix_csv_path, index=False)
    print(f"[CROSS] Connectivity matrix written to {matrix_csv_path}", file=sys.stderr)

    # Node-level cross-subdomain stats
    print("[CROSS] Computing node-level cross-subdomain stats ...", file=sys.stderr)
    node_stats_df = compute_node_cross_subdomain_stats(G)
    node_stats_csv_path = os.path.join(
        args.output_dir, "cross_subdomain_node_stats.csv"
    )
    node_stats_df.to_csv(node_stats_csv_path, index=False)
    print(f"[CROSS] Node stats written to {node_stats_csv_path}", file=sys.stderr)

    # Communities & subdomain mixing
    print(
        "[COMMUNITIES] Detecting global communities and subdomain mix ...",
        file=sys.stderr,
    )
    communities_mix_df, node_labels_df = detect_global_communities_and_subdomain_mix(G)
    if not communities_mix_df.empty:
        comm_mix_csv_path = os.path.join(
            args.output_dir, "cross_communities_subdomain_mix.csv"
        )
        communities_mix_df.to_csv(comm_mix_csv_path, index=False)
        print(
            f"[COMMUNITIES] Community subdomain mix written to {comm_mix_csv_path}",
            file=sys.stderr,
        )

        comm_labels_csv_path = os.path.join(
            args.output_dir, "cross_community_node_labels.csv"
        )
        node_labels_df.to_csv(comm_labels_csv_path, index=False)
        print(
            f"[COMMUNITIES] Community node labels written to {comm_labels_csv_path}",
            file=sys.stderr,
        )
    else:
        print("[COMMUNITIES] No communities detected (graph may be empty).", file=sys.stderr)

    # High-level summary JSON
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_cross_edges = 0
    num_within_edges = 0
    for u, v in G.edges():
        su = G.nodes[u].get("primary_subdomain", "unknown")
        sv = G.nodes[v].get("primary_subdomain", "unknown")
        if su == sv:
            num_within_edges += 1
        else:
            num_cross_edges += 1

    summary = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "subdomain_node_counts": subdomain_node_counts,
        "num_cross_edges": num_cross_edges,
        "num_within_edges": num_within_edges,
    }
    summary_path = os.path.join(args.output_dir, "cross_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] Cross-subdomain summary written to {summary_path}", file=sys.stderr)

    # Plot subdomain connectivity matrix
    print("[PLOT] Generating subdomain connectivity heatmap ...", file=sys.stderr)
    matrix_plot_path = os.path.join(
        args.output_dir, "subdomain_connectivity_matrix.png"
    )
    plot_subdomain_matrix(matrix_df, matrix_plot_path)
    print(f"[PLOT] Heatmap written to {matrix_plot_path}", file=sys.stderr)

    # Plot a core node–edge graph of the most important repositories
    print("[PLOT] Generating core network graph ...", file=sys.stderr)
    core_graph_path = os.path.join(args.output_dir, "core_network_graph.png")
    if not node_stats_df.empty and not node_labels_df.empty:
        plot_core_network_graph(G, node_stats_df, node_labels_df, core_graph_path)
        print(f"[PLOT] Core network graph written to {core_graph_path}", file=sys.stderr)
    else:
        print("[PLOT] Skipping core network graph (missing stats or labels).", file=sys.stderr)

    print("[DONE] Cross-subdomain analysis complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
