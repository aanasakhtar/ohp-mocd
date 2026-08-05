"""
config.py — Central configuration for all OHP-MOCD experiments & benchmark datasets.
"""

import os

# HP-MOCD baseline hyperparameters
HPMOCD_CONFIG = {
    "population_size": 100,
    "max_generations": 100,
    "crossover_prob": 0.85,
    "mutation_prob": 0.35,
    "seed": None,
}

# LFR Benchmark parameters
LFR_CONFIG = {
    "n": 1000,
    "tau1": 2.5,
    "tau2": 1.5,
    "mu": 0.3,
    "average_degree": 20,
    "max_degree": 50,
    "min_community": 20,
    "max_community": 100,
    "overlap_n": 200,
    "overlap_membership": 2,
    "seed": None,
}

# DBLP Co-authorship dataset
DBLP_CONFIG = {
    "url_graph": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
    "url_cmty": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
    "save_dir": "data/dblp_raw/",
    "subsample_nodes": 10_000,
    "seed": 42,
}

# Amazon co-purchasing dataset
AMAZON_CONFIG = {
    "url_graph": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
    "url_cmty": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.dedup.cmty.txt.gz",
    "save_dir": "data/amazon_raw/",
    "subsample_nodes": 10_000,
    "seed": 42,
}

# Facebook ego-network dataset (Full Network, 4,039 nodes)
FACEBOOK_CONFIG = {
    "url_graph": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "url_cmty": "https://snap.stanford.edu/data/facebook.tar.gz",
    "save_dir": "data/facebook_raw/",
    "subsample_nodes": None, # Full network
    "seed": 42,
}

# Youtube user groups dataset
YOUTUBE_CONFIG = {
    "url_graph": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
    "url_cmty": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz",
    "save_dir": "data/youtube_raw/",
    "subsample_nodes": 10_000,
    "seed": 42,
}
