"""graph_analytics.py — Graph metrics for GEO-Digest knowledge graph.

Pure-Python implementation of centrality and community detection algorithms.
No external dependencies (no networkx) — works in Docker with minimal image.

Algorithms:
  - PageRank: iterative eigenvalue method
  - Betweenness centrality: Brandes algorithm (O(VE) for unweighted)
  - Louvain community detection: modularity optimization

Usage:
    from graph_analytics import compute_all_metrics
    metrics = compute_all_metrics(nodes, edges)
    # → {page_rank: {...}, betweenness: {...}, communities: {...}}
"""

from collections import defaultdict, deque


# ════════════════════════════════════════════════
#  PageRank
# ════════════════════════════════════════════════

def compute_pagerank(
    nodes: list[str],
    edges: list[tuple[str, str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank scores for nodes.

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples (undirected)
        damping: probability of following a link (default 0.85)
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        Dict mapping node_id → PageRank score (sum ≈ 1.0)
    """
    if not nodes or not edges:
        return {n: 1.0 / len(nodes) if nodes else 0.0 for n in nodes}

    n = len(nodes)
    node_set = set(nodes)

    # Build adjacency list (undirected)
    out_links = defaultdict(list)
    for src, tgt in edges:
        if src in node_set and tgt in node_set:
            out_links[src].append(tgt)
            out_links[tgt].append(src)  # undirected

    # Handle dangling nodes (no outgoing links)
    dangling = [node for node in nodes if not out_links.get(node)]

    # Initialize uniformly
    pr = {node: 1.0 / n for node in nodes}

    for iteration in range(max_iter):
        new_pr = {}
        dangling_sum = sum(pr[d] for d in dangling)

        for node in nodes:
            # Sum of incoming PageRank contributions
            rank_sum = sum(
                pr[neighbor] / len(out_links[neighbor])
                for neighbor in out_links[node]
                if neighbor != node
            )
            new_pr[node] = (1 - damping) / n + damping * (rank_sum + dangling_sum / n)

        # Check convergence
        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr
        if diff < tol:
            print(f"  [PageRank] Converged at iteration {iteration+1} (diff={diff:.2e})")
            break

    # Normalize to [0, 1] range
    max_val = max(pr.values()) if pr.values() else 1.0
    if max_val > 0:
        pr = {k: v / max_val for k, v in pr.items()}
    else:
        pr = {k: 0.0 for k in pr}

    return pr


# ════════════════════════════════════════════════
#  Betweenness Centrality (Brandes algorithm)
# ════════════════════════════════════════════════

def compute_betweenness(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute betweenness centrality using Brandes algorithm.

    For each node, counts the fraction of shortest paths
    that pass through that node.

    Complexity: O(VE) for unweighted graphs.
    """
    if not nodes:
        return {}

    node_set = set(nodes)

    # Build adjacency list
    adj = defaultdict(set)
    for src, tgt in edges:
        if src in node_set and tgt in node_set and src != tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)

    # Initialize
    BC = {n: 0.0 for n in nodes}

    for s in nodes:
        # BFS tree from source s
        stack = []
        predecessors = defaultdict(list)  # w → list of v that have w as predecessor
        sigma = {n: 0 for n in nodes}     # number of shortest paths
        sigma[s] = 1
        dist = {s: 0}
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                # First time visiting w?
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                # Shortest path to w via v?
                if dist.get(w, float('inf')) == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Accumulation — back-propagation of dependencies
        delta = {n: 0.0 for n in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                BC[w] += delta[w]

    # Normalize by dividing by (n-1)*(n-2) for undirected graphs
    n = len(nodes)
    norm = ((n - 1) * (n - 2)) / 2 if n > 2 else 1.0
    if norm > 0:
        BC = {k: v / norm for k, v in BC.items()}

    return BC


# ════════════════════════════════════════════════
#  Louvain Community Detection
# ════════════════════════════════════════════════

def detect_communities_louvain(
    nodes: list[str],
    edges: list[tuple[str, str]],
    weight_fn=None,
    max_passes: int = 20,
) -> dict[str, int]:
    """Detect communities using Louvain modularity optimization.

    Simple greedy modularity maximization:
      Phase 1: Each node moves to the community that maximizes modularity gain
      Phase 2: Aggregate communities into super-nodes, repeat

    Args:
        nodes: list of node IDs
        edges: list of (source, target) tuples
        weight_fn: optional function (src, tgt) → float edge weight
        max_passes: maximum number of passes through all nodes

    Returns:
        Dict mapping node_id → community_id (int)
    """
    if not nodes:
        return {}

    node_set = set(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Build weighted adjacency
    adj = defaultdict(dict)  # node → {neighbor: weight}
    total_weight = 0.0

    for src, tgt in edges:
        if src in node_set and tgt in node_set:
            w = weight_fn(src, tgt) if weight_fn else 1.0
            if tgt in adj[src]:
                adj[src][tgt] += w
            else:
                adj[src][tgt] = w
            if src in adj[tgt]:
                adj[tgt][src] += w
            else:
                adj[tgt][src] = w
            total_weight += w * 2  # count both directions

    if total_weight == 0:
        return {n: i for i, n in enumerate(nodes)}

    m = total_weight  # sum of all edge weights (2*actual for undirected)

    # Initialize: each node in its own community
    node_to_comm = {n: i for i, n in enumerate(nodes)}
    comm_nodes = {i: [n] for i, n in enumerate(nodes)}

    # Compute weighted degree (sum of edge weights for each node)
    k = {n: sum(adj[n].values()) for n in nodes}

    # Sum of weights inside each community + total degree per community
    def _comm_total_degree(comm_id):
        return sum(k[n] for n in comm_nodes.get(comm_id, []))

    def _comm_internal_weight(comm_id):
        w = 0.0
        members = comm_nodes.get(comm_id, [])
        member_set = set(members)
        for n in members:
            for nbr, wt in adj[n].items():
                if nbr in member_set and member_set.index(nbr) > members.index(n):
                    w += wt
        return w

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        nodes_shuffled = list(nodes)
        # Deterministic shuffle for reproducibility
        nodes_shuffled.sort()

        for node in nodes_shuffled:
            best_comm = node_to_comm[node]
            best_gain = 0.0

            comm_of_node = node_to_comm[node]
            k_i = k.get(node, 0)
            ki_in = sum(
                adj[node].get(nbr, 0)
                for nbr in adj.get(node, {})
                if node_to_comm.get(nbr) == comm_of_node
            )
            sigma_tot = _comm_total_degree(comm_of_node)

            # Find neighboring communities
            neighbor_comms = set()
            for nbr in adj.get(node, {}):
                if nbr != node:
                    neighbor_comms.add(node_to_comm[nbr])

            # Try moving to each neighboring community
            for target_comm in sorted(neighbor_comms):
                if target_comm == comm_of_node:
                    continue

                sigma_tot_target = _comm_total_degree(target_comm)
                ki_in_target = sum(
                    adj[node].get(nbr, 0)
                    for nbr in adj.get(node, {})
                    if node_to_comm.get(nbr) == target_comm
                )

                delta_q = (
                    (ki_in_target - ki_in) / m
                    + k_i * (sigma_tot - sigma_tot_target - k_i) / (2 * m * m)
                )

                if delta_q > best_gain:
                    best_gain = delta_q
                    best_comm = target_comm

            # Move node if improvement found
            if best_gain > 1e-10 and best_comm != comm_of_node:
                # Remove from old community
                if comm_of_node in comm_nodes:
                    comm_nodes[comm_of_node].remove(node)
                    if not comm_nodes[comm_of_node]:
                        del comm_nodes[comm_of_node]

                # Add to new community
                node_to_comm[node] = best_comm
                if best_comm not in comm_nodes:
                    comm_nodes[best_comm] = []
                comm_nodes[best_comm].append(node)
                improved = True

    # Renumber communities to be contiguous starting from 0
    unique_comms = sorted(set(node_to_comm.values()))
    comm_remap = {old: new for new, old in enumerate(unique_comms)}
    return {n: comm_remap[c] for n, c in node_to_comm.items()}


# ════════════════════════════════════════════════
#  Combined pipeline
# ════════════════════════════════════════════════

def compute_all_metrics(
    nodes: list[dict],
    edges: list[dict],
    article_only: bool = True,
) -> dict:
    """Compute all graph metrics and annotate nodes.

    Args:
        nodes: list of Cytoscape-style {"data": {...}} nodes
        edges: list of Cytoscape-style {"data": {...}} edges
        article_only: if True, only compute metrics on article subgraph

    Returns:
        {
            "page_rank": {node_id: float},
            "betweenness": {node_id: float},
            "communities": {node_id: int},
            "community_info": [{id, size, labels}],
            "hub_nodes": [node_id],       # top 5% by PageRank
            "bridge_nodes": [node_id],    # top 5% by betweenness
        }
    """
    # Filter to article nodes only for meaningful metrics
    if article_only:
        art_ids = {n["data"]["id"] for n in nodes if n["data"].get("nodeType") == "article"}
        filtered_nodes = [n["data"]["id"] for n in nodes if n["data"].get("nodeType") == "article"]
    else:
        art_ids = {n["data"]["id"] for n in nodes}
        filtered_nodes = [n["data"]["id"] for n in nodes]

    # Extract edge tuples (only article-to-article)
    edge_tuples = []
    for e in edges:
        d = e["data"]
        s, t = d.get("source", ""), d.get("target", "")
        if s in art_ids and t in art_ids and s != t:
            conf = d.get("confidence", 1.0)
            edge_tuples.append((s, t))

    if len(filtered_nodes) < 2:
        empty = {n: 0.0 for n in filtered_nodes}
        return {
            "page_rank": empty,
            "betweenness": empty,
            "communities": {n: 0 for n in filtered_nodes},
            "community_info": [],
            "hub_nodes": [],
            "bridge_nodes": [],
        }

    print(f"  [Analytics] Computing metrics on {len(filtered_nodes)} nodes, {len(edge_tuples)} edges")

    # PageRank
    pr = compute_pagerank(filtered_nodes, edge_tuples)
    print(f"  [PageRank] Done. Top 3: {sorted(pr.items(), key=lambda x:-x[1])[:3]}")

    # Betweenness
    bc = compute_betweenness(filtered_nodes, edge_tuples)
    print(f"  [Betweenness] Done. Top 3: {sorted(bc.items(), key=lambda x:-x[1])[:3]}")

    # Communities (Louvain)
    comms = detect_communities_louvain(filtered_nodes, edge_tuples)
    comm_counts = defaultdict(int)
    comm_labels = defaultdict(list)
    for nid, cid in comms.items():
        comm_counts[cid] += 1
        # Get label for this node
        label = "?"
        for n in nodes:
            if n["data"]["id"] == nid:
                label = n["data"].get("label", "?")[:40]
                break
        comm_labels[cid].append(label)

    num_comms = len(comm_counts)
    print(f"  [Communities] Found {num_comms} communities: "
          f"{dict(sorted(comm_counts.items(), key=lambda x:-x[1]))}")

    # Identify hubs (top 5% by PageRank) and bridges (top 5% by betweenness)
    n_top = max(1, len(filtered_nodes) // 20)  # top 5%
    sorted_pr = sorted(pr.items(), key=lambda x: -x[1])
    sorted_bc = sorted(bc.items(), key=lambda x: -x[1])
    hub_nodes = [nid for nid, _ in sorted_pr[:n_top]]
    bridge_nodes = [nid for nid, _ in sorted_bc[:n_top]]

    community_info = [
        {
            "id": cid,
            "size": cnt,
            "labels": comm_labels[cid][:5],
            "label_count": len(comm_labels[cid]),
        }
        for cid, cnt in sorted(comm_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "page_rank": pr,
        "betweenness": bc,
        "communities": comms,
        "community_info": community_info,
        "hub_nodes": hub_nodes,
        "bridge_nodes": bridge_nodes,
    }
