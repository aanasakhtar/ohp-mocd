"""
data/load_facebook.py — Download and load SNAP ego-Facebook social circles dataset.
Source: https://snap.stanford.edu/data/ego-Facebook.html
"""

import gzip
import tarfile
import sys
import urllib.request
from pathlib import Path
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FACEBOOK_CONFIG


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already cached: {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def load_facebook(cfg: dict = FACEBOOK_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    graph_gz = save_dir / "facebook_combined.txt.gz"
    circles_tar = save_dir / "facebook.tar.gz"

    graph_url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    circles_url = "https://snap.stanford.edu/data/facebook.tar.gz"

    _download(graph_url, graph_gz)
    _download(circles_url, circles_tar)

    print("Parsing Facebook edge list ...")
    G = nx.Graph()
    with gzip.open(graph_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    print(f"  Full Facebook graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_set = set(G.nodes())
    communities: list[frozenset] = []

    print("Parsing Facebook social circles ground-truth ...")
    with tarfile.open(circles_tar, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".circles"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in f.read().decode("utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) > 1:
                        members = frozenset(int(x) for x in parts[1:]) & node_set
                        if len(members) >= 3:
                            communities.append(members)

    overlapping = sum(1 for v in node_set if sum(v in c for c in communities) > 1)
    print(f"  Social Circles (Ground-Truth): {len(communities)}")
    print(f"  Overlapping nodes: {overlapping} ({100*overlapping/len(node_set):.1f}%)")

    return G, communities


if __name__ == "__main__":
    G, cmty = load_facebook()
    print(f"Ready: Facebook graph has {G.number_of_nodes()} nodes, {len(cmty)} communities.")
