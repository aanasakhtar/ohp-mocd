"""
data/load_dblp.py — Download and load the SNAP DBLP co-authorship network.
Source: https://snap.stanford.edu/data/com-DBLP.html
"""

import gzip
import random
import sys
import urllib.request
from pathlib import Path
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DBLP_CONFIG


def _resolve_working_url(candidates: list[str]) -> str:
    last_error: Exception | None = None
    for url in candidates:
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=20):
                return url
        except Exception as e:
            last_error = e
    if last_error is None:
        raise RuntimeError("No DBLP URL candidates provided")
    raise last_error


def _download(url: str, dest: Path) -> None:
    def _is_gzip_ok(path: Path) -> bool:
        try:
            with gzip.open(path, "rb") as fh:
                for _ in iter(lambda: fh.read(1 << 20), b""):
                    pass
            return True
        except Exception:
            return False

    if dest.exists() and _is_gzip_ok(dest):
        print(f"  Already cached: {dest}")
        return

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def load_dblp(cfg: dict = DBLP_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    graph_gz = save_dir / "com-DBLP.ungraph.txt.gz"
    cmty_gz = save_dir / "com-DBLP.all.cmty.txt.gz"

    graph_candidates = [
        cfg.get("url_graph", ""),
        "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
    ]
    cmty_candidates = [
        cfg.get("url_cmty", ""),
        "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
    ]

    graph_url = _resolve_working_url([u for u in graph_candidates if u])
    cmty_url = _resolve_working_url([u for u in cmty_candidates if u])

    _download(graph_url, graph_gz)
    _download(cmty_url, cmty_gz)

    print("Parsing DBLP edge list ...")
    G = nx.Graph()
    with gzip.open(graph_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    print(f"  Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    subsample = cfg.get("subsample_nodes")
    if subsample and G.number_of_nodes() > subsample:
        rng = random.Random(cfg["seed"])
        start = rng.choice(list(G.nodes()))
        bfs_nodes = list(nx.bfs_tree(G, start).nodes())[:subsample]
        G = G.subgraph(bfs_nodes).copy()
        print(f"  Subsampled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_set = set(G.nodes())
    communities: list[frozenset] = []

    print("Parsing DBLP community file ...")
    with gzip.open(cmty_gz, "rb") as f:
        buf = b""
        for chunk in iter(lambda: f.read(1 << 16), b""):
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                line = buf[:nl]
                buf = buf[nl + 1 :]
                if not line:
                    continue
                try:
                    members = frozenset(int(x) for x in line.split()) & node_set
                    if len(members) >= 3:
                        communities.append(members)
                except Exception:
                    continue
        if buf:
            try:
                members = frozenset(int(x) for x in buf.split()) & node_set
                if len(members) >= 3:
                    communities.append(members)
            except Exception:
                pass

    overlapping = sum(1 for v in node_set if sum(v in c for c in communities) > 1)
    print(f"  Communities (in subgraph): {len(communities)}")
    print(f"  Overlapping nodes: {overlapping} ({100*overlapping/len(node_set):.1f}%)")

    return G, communities


if __name__ == "__main__":
    G, cmty = load_dblp()
    print(f"Ready: DBLP graph has {G.number_of_nodes()} nodes, {len(cmty)} communities.")
