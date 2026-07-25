"""
data/config.py — Benchmark configurations for LFR and DBLP graph generators.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "dblp_raw"

LFR_CONFIG = {
    "n": 250,
    "tau1": 3.0,
    "tau2": 1.5,
    "mu": 0.1,
    "average_degree": 5,
    "max_degree": 15,
    "min_community": 10,
    "max_community": 30,
    "seed": 42,
    "overlap_n": 50,
    "overlap_membership": 2,
}

DBLP_CONFIG = {
    "save_dir": str(DATA_DIR),
    "subsample_nodes": 500,
    "seed": 42,
    "url_graph": "https://snap.stanford.edu/data/com-DBLP.ungraph.txt.gz",
    "url_cmty": "https://snap.stanford.edu/data/com-DBLP.all.cmty.txt.gz",
}
