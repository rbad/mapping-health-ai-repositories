#!/usr/bin/env python3
"""
Analyze health-AI GitHub repositories for one subdomain by building a repository-level
collaboration network, computing graph statistics, and detecting communities.

Usage example:
    python analyze_health_ai_subdomain_network.py --input repos_health_ai_imaging.jsonl --output-dir imaging_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_comm
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------- Data loading ----------

def load_repos(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load repositories from a JSONL file into a list of dicts."""
    repos: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            repos.append(json.loads(line))
    return repos


# ---------- Graph construction ----------

def is_bot_contributor(login: str) -> bool:
    """Heuristic to ignore obvious bots / automation accounts."""
    if not login:
        return False
    name = login.lower()
    if "[bot]" in name:
        return True
    if name.endswith("bot"):
        return True
    # Some known automation accounts seen in data
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


def build_repo_graph(
    repos: Sequence[Dict[str, Any]],
    min_shared_contributors: int = 1,
    include_parent_fork_edges: bool = True,
) -> Tuple[nx.Graph, Dict[int, Dict[str, Any]]]:
    """
    Build an undirected graph where nodes are repositories and edges
    represent collaboration via shared contributors and optional parent–fork ties.
    """
    G = nx.Graph()
    id_to_repo: Dict[int, Dict[str, Any]] = {}
    full_name_to_id: Dict[str, int] = {}

    # Add nodes with metadata
    for rec in repos:
        rid = rec["repo_id"]
        id_to_repo[rid] = rec
        full_name_to_id[rec["full_name"]] = rid

        G.add_node(
            rid,
            full_name=rec.get("full_name"),
            name=rec.get("name"),
            description=rec.get("description"),
            language=rec.get("language"),
            topics=",".join(rec.get("topics") or []),
            stargazers_count=rec.get("stargazers_count", 0),
            watchers_count=rec.get("watchers_count", 0),
            open_issues_count=rec.get("open_issues_count", 0),
            forks_count=rec.get("forks_count", 0),
            is_fork=rec.get("is_fork", False),
            license_spdx_id=rec.get("license_spdx_id"),
            license_name=rec.get("license_name"),
            health_subdomain=rec.get("health_subdomain"),
        )

    # Map contributors to the repos they contributed to
    contrib_to_repos: Dict[str, Set[int]] = defaultdict(set)
    for rec in repos:
        rid = rec["repo_id"]
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
        # For each contributor, connect every pair of repos they touched
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a == b:
                    continue
                edge_weights[(a, b)] += 1

    # Add edges to graph
    for (a, b), w in edge_weights.items():
        if w < min_shared_contributors:
            continue
        if G.has_edge(a, b):
            # Should not normally happen, but be robust
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

    # Optional: add parent–fork edges if parent_full_name is available in data
    if include_parent_fork_edges:
        for rec in repos:
            rid = rec["repo_id"]
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


# ---------- Network-level stats ----------

def compute_network_summary(G: nx.Graph) -> Dict[str, Any]:
    """Compute basic network-level statistics."""

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    components = list(nx.connected_components(G))
    component_sizes = sorted((len(c) for c in components), reverse=True)
    num_components = len(component_sizes)
    largest_component_size = component_sizes[0] if component_sizes else 0

    degrees = [d for _, d in G.degree()]
    weighted_degrees = [d for _, d in G.degree(weight="weight")]

    def safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    def safe_median(xs):
        n = len(xs)
        if n == 0:
            return 0.0
        xs_sorted = sorted(xs)
        mid = n // 2
        if n % 2 == 1:
            return float(xs_sorted[mid])
        return float(0.5 * (xs_sorted[mid - 1] + xs_sorted[mid]))

    summary = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_components": num_components,
        "component_sizes": component_sizes,
        "largest_component_size": largest_component_size,
        "avg_degree": safe_mean(degrees),
        "median_degree": safe_median(degrees),
        "avg_weighted_degree": safe_mean(weighted_degrees),
        "median_weighted_degree": safe_median(weighted_degrees),
    }

    return summary


# ---------- Node-level metrics ----------

def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute per-node metrics:
    - degree, weighted degree
    - component id
    - degree centrality, betweenness centrality, PageRank
    (community_id is added later by merging with community labels)
    """
    # Connected components and component IDs
    components = list(nx.connected_components(G))
    components_sorted = sorted(components, key=len, reverse=True)
    node_to_component: Dict[int, int] = {}
    for cid, comp in enumerate(components_sorted):
        for nid in comp:
            node_to_component[nid] = cid

    largest_comp_nodes: Set[int] = set(components_sorted[0]) if components_sorted else set()
    G_largest = G.subgraph(largest_comp_nodes).copy() if largest_comp_nodes else nx.Graph()

    # Degree metrics
    degree_dict = dict(G.degree())
    weighted_degree_dict = dict(G.degree(weight="weight"))

    # Centrality on largest component only (for speed)
    if G_largest.number_of_nodes() > 0:
        deg_centrality = nx.degree_centrality(G_largest)
        # Approximate betweenness with sampling to keep runtime manageable
        k_sample = min(500, G_largest.number_of_nodes())
        betw_centrality = nx.betweenness_centrality(
            G_largest,
            k=k_sample,
            seed=42,
            weight="weight",
        )
        pagerank = nx.pagerank(G_largest, weight="weight")
    else:
        deg_centrality = {}
        betw_centrality = {}
        pagerank = {}

    records: List[Dict[str, Any]] = []
    for nid in G.nodes():
        attrs = G.nodes[nid]
        record: Dict[str, Any] = {
            "repo_id": nid,
            "full_name": attrs.get("full_name"),
            "name": attrs.get("name"),
            "language": attrs.get("language"),
            "topics": attrs.get("topics"),
            "stargazers_count": attrs.get("stargazers_count", 0),
            "watchers_count": attrs.get("watchers_count", 0),
            "open_issues_count": attrs.get("open_issues_count", 0),
            "forks_count": attrs.get("forks_count", 0),
            "license_spdx_id": attrs.get("license_spdx_id"),
            "license_name": attrs.get("license_name"),
            "health_subdomain": attrs.get("health_subdomain"),
            "degree": degree_dict.get(nid, 0),
            "weighted_degree": float(weighted_degree_dict.get(nid, 0.0)),
            "component_id": node_to_component.get(nid, -1),
            "in_largest_component": nid in largest_comp_nodes,
            "degree_centrality": float(deg_centrality.get(nid, 0.0)),
            "betweenness_centrality": float(betw_centrality.get(nid, 0.0)),
            "pagerank": float(pagerank.get(nid, 0.0)),
        }
        records.append(record)

    return pd.DataFrame.from_records(records)


# ---------- Community detection + labeling ----------

def detect_communities_and_label(
    G: nx.Graph,
    id_to_repo: Dict[int, Dict[str, Any]],
    max_terms: int = 10,
    min_df: int = 5,
    max_features: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform community detection on the largest connected component and
    label each community using TF-IDF over descriptions + README text.

    Returns:
        - communities_df: one row per community
        - node_labels_df: repo_id -> community_id for nodes in largest component
    """
    components = list(nx.connected_components(G))
    if not components:
        return pd.DataFrame(), pd.DataFrame()

    components_sorted = sorted(components, key=len, reverse=True)
    largest_comp_nodes: Set[int] = set(components_sorted[0])
    G_largest = G.subgraph(largest_comp_nodes).copy()

    # Louvain communities
    communities = nx_comm.louvain_communities(
        G_largest,
        weight="weight",
        seed=42,
    )

    # Prepare documents (description + README) for TF-IDF
    node_list: List[int] = list(G_largest.nodes())
    node_index: Dict[int, int] = {nid: i for i, nid in enumerate(node_list)}
    docs: List[str] = []
    for nid in node_list:
        rec = id_to_repo.get(nid, {})
        text = (rec.get("description") or "") + " " + (rec.get("readme_text") or "")
        docs.append(text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.5,
        min_df=min_df,
        max_features=max_features,
    )
    tfidf = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    community_records: List[Dict[str, Any]] = []
    node_label_records: List[Dict[str, Any]] = []

    for comm_id, comm_nodes in enumerate(communities):
        idxs = [node_index[nid] for nid in comm_nodes]
        if not idxs:
            continue
        submatrix = tfidf[idxs, :]
        scores = np.asarray(submatrix.sum(axis=0)).ravel()
        if scores.size == 0:
            top_terms: List[str] = []
        else:
            top_indices = scores.argsort()[::-1][:max_terms]
            top_terms = [terms[i] for i in top_indices if scores[i] > 0]

        # Example repositories: top few by degree within this community
        degrees_in_comm = G_largest.degree(comm_nodes)
        top_nodes = sorted(
            degrees_in_comm,
            key=lambda x: (-x[1], x[0]),
        )[:5]
        top_repo_names = [G_largest.nodes[nid].get("full_name") for nid, _ in top_nodes]

        community_records.append(
            {
                "community_id": comm_id,
                "size": len(comm_nodes),
                "top_terms": ", ".join(top_terms),
                "example_repos": ", ".join(top_repo_names),
            }
        )

        for nid in comm_nodes:
            node_label_records.append(
                {"repo_id": nid, "community_id": comm_id}
            )

    communities_df = pd.DataFrame.from_records(community_records)
    node_labels_df = pd.DataFrame.from_records(node_label_records)

    return communities_df, node_labels_df


# ---------- Edge export ----------

def export_edge_list(G: nx.Graph) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        records.append(
            {
                "source_repo_id": int(u),
                "target_repo_id": int(v),
                "weight": float(data.get("weight", 1.0)),
                "shared_contributors": int(data.get("shared_contributors", 0)),
                "parent_fork": bool(data.get("parent_fork", False)),
            }
        )
    return pd.DataFrame.from_records(records)


# ---------- Plotting helpers ----------

def plot_degree_distribution(G: nx.Graph, output_path: str) -> None:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return
    plt.figure()
    plt.hist(degrees, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Count of repositories")
    plt.title("Degree distribution (repository collaboration network)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_component_size_distribution(component_sizes, output_path: str) -> None:
    if not component_sizes:
        return
    plt.figure()
    plt.hist(component_sizes, bins=50)
    plt.xlabel("Component size (number of repositories)")
    plt.ylabel("Count of components")
    plt.title("Connected component size distribution")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_community_sizes(communities_df: pd.DataFrame, output_path: str, top_n: int = 20) -> None:
    if communities_df.empty:
        return
    df = communities_df.sort_values("size", ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(df["community_id"].astype(str), df["size"])
    plt.xlabel("Community ID")
    plt.ylabel("Number of repositories")
    plt.title(f"Top {top_n} communities by size")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------- Main CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and analyze a repository collaboration network from JSONL data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file (e.g., repos_health_ai_imaging.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where analysis outputs (CSVs, JSON, PNGs) will be written.",
    )
    parser.add_argument(
        "--min-shared-contributors",
        type=int,
        default=1,
        help="Minimum number of shared contributors required to create an edge.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[LOAD] Reading repositories from {args.input} ...")
    repos = load_repos(args.input)
    print(f"[LOAD] Loaded {len(repos)} repositories.")

    print("[GRAPH] Building collaboration network ...")
    G, id_to_repo = build_repo_graph(
        repos,
        min_shared_contributors=args.min_shared_contributors,
        include_parent_fork_edges=True,
    )
    print(
        f"[GRAPH] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    print("[STATS] Computing network summary ...")
    summary = compute_network_summary(G)
    summary_path = os.path.join(args.output_dir, "network_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[STATS] Summary written to {summary_path}")

    print("[NODES] Computing node-level metrics ...")
    nodes_df = compute_node_metrics(G)

    print("[COMMUNITIES] Detecting communities and labeling them ...")
    communities_df, node_labels_df = detect_communities_and_label(G, id_to_repo)
    if not communities_df.empty:
        communities_csv_path = os.path.join(args.output_dir, "communities_summary.csv")
        communities_df.to_csv(communities_csv_path, index=False)
        print(f"[COMMUNITIES] Community summary written to {communities_csv_path}")

        node_labels_csv_path = os.path.join(args.output_dir, "node_community_labels.csv")
        node_labels_df.to_csv(node_labels_csv_path, index=False)
        print(f"[COMMUNITIES] Node community labels written to {node_labels_csv_path}")

        # Merge community_id into node metrics
        nodes_df = nodes_df.merge(
            node_labels_df,
            on="repo_id",
            how="left",
        )
        nodes_df["community_id"] = nodes_df["community_id"].fillna(-1).astype(int)
    else:
        print("[COMMUNITIES] No communities detected (graph may be empty).")
        nodes_df["community_id"] = -1

    nodes_csv_path = os.path.join(args.output_dir, "nodes_with_metrics.csv")
    nodes_df.to_csv(nodes_csv_path, index=False)
    print(f"[NODES] Node metrics written to {nodes_csv_path}")

    print("[EDGES] Exporting edge list ...")
    edges_df = export_edge_list(G)
    edges_csv_path = os.path.join(args.output_dir, "edges.csv")
    edges_df.to_csv(edges_csv_path, index=False)
    print(f"[EDGES] Edge list written to {edges_csv_path}")

    # Figures
    print("[PLOTS] Generating figures ...")
    deg_plot_path = os.path.join(args.output_dir, "degree_distribution.png")
    plot_degree_distribution(G, deg_plot_path)
    comp_plot_path = os.path.join(args.output_dir, "component_size_distribution.png")
    plot_component_size_distribution(summary.get("component_sizes", []), comp_plot_path)
    if not communities_df.empty:
        comm_plot_path = os.path.join(args.output_dir, "community_sizes.png")
        plot_community_sizes(communities_df, comm_plot_path)

    print("[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
