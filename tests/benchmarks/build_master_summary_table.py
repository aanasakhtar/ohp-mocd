"""
build_master_summary_table.py

Builds the Master Unified Comparative Table across all 4 baseline papers in the exact hierarchy:
  Baseline Paper Algorithm (X) -> Dataset -> Metric -> Side-by-Side Scores -> Absolute Diff (Delta) -> Pct Improvement (%) -> Outperforming Algorithm
"""

import os
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

df1 = pd.read_csv(BENCH_DIR / "strict_paper1_slpa_qov.csv")
df2 = pd.read_csv(BENCH_DIR / "strict_paper2_mcmoea_qov.csv")
df3 = pd.read_csv(BENCH_DIR / "strict_paper3_fccni_eq.csv")
df4 = pd.read_csv(BENCH_DIR / "strict_paper4_cetin_q_coverage.csv")

master_rows = []

# 1. Algorithm X = SLPA (2011)
for _, r in df1.iterrows():
    b_val = r['OHP_MOCD_BoundarySeeded_Qov']
    c_val = r['OHP_MOCD_Crisp_Qov']
    best_ohp = max(b_val, c_val)
    base = r['SLPA_Qov_Reported']
    diff = best_ohp - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp > base else 'SLPA'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'SLPA (2011)',
        'Dataset': f"{r['Dataset']} (N = {r['Nodes']})",
        'Metric': 'Nicosia Qov',
        'Algorithm X Reported Score': f"{base:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{b_val:.4f}",
        'OHP-MOCD (Crisp)': f"{c_val:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp:.4f}",
        'Absolute Diff (Delta)': f"{diff:+.4f}",
        'Pct Improvement (%)': f"{pct:+.2f}%",
        'Outperforming Algorithm': winner
    })

# 2. Algorithm X = MCMOEA (2016)
for _, r in df2.iterrows():
    b_val = r['OHP_MOCD_BoundarySeeded_Qov']
    c_val = r['OHP_MOCD_Crisp_Qov']
    best_ohp = max(b_val, c_val)
    base = r['MCMOEA_Qov_Reported']
    diff = best_ohp - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp > base else 'MCMOEA'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'MCMOEA (2016)',
        'Dataset': r['Dataset'],
        'Metric': 'Nicosia Qov',
        'Algorithm X Reported Score': f"{base:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{b_val:.4f}",
        'OHP-MOCD (Crisp)': f"{c_val:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp:.4f}",
        'Absolute Diff (Delta)': f"{diff:+.4f}",
        'Pct Improvement (%)': f"{pct:+.2f}%",
        'Outperforming Algorithm': winner
    })

# 3. Algorithm X = FCCNI (2024) - Direct gNMI Comparison
for _, r in df3.iterrows():
    net = r['Dataset']
    b_gnmi = r['OHP_MOCD_BoundarySeeded_gNMI']
    c_gnmi = r['OHP_MOCD_Crisp_gNMI']
    best_ohp_g = max(b_gnmi, c_gnmi)
    base = r['FCCNI_gNMI_max']
    diff = best_ohp_g - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp_g >= base else 'FCCNI'
    
    n_map = {"Karate": "N = 34", "Dolphins": "N = 62", "Polbooks": "N = 105", "Football": "N = 115"}
    n_str = f"{net} ({n_map.get(net, '')})" if net in n_map else net
    
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'FCCNI (2024)',
        'Dataset': n_str,
        'Metric': 'gNMI',
        'Algorithm X Reported Score': f"{base:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{b_gnmi:.4f}",
        'OHP-MOCD (Crisp)': f"{c_gnmi:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp_g:.4f}",
        'Absolute Diff (Delta)': f"{diff:+.4f}",
        'Pct Improvement (%)': f"{pct:+.2f}%",
        'Outperforming Algorithm': winner
    })

# 4. Algorithm X = Çetin & Amrahov (2022)
for _, r in df4.iterrows():
    net = r['Dataset']
    n_map = {"Karate": "N = 34", "Dolphins": "N = 62", "Lesmis": "N = 77", "Polbooks": "N = 105"}
    n_str = f"{net} ({n_map.get(net, '')})" if net in n_map else net
    
    # Shen Q
    b_q = r['OHP_MOCD_BoundarySeeded_Shen_Q']
    c_q = r['OHP_MOCD_Crisp_Shen_Q']
    best_ohp_q = max(b_q, c_q)
    base_q = r['Proposed_Cetin_Shen_Q']
    diff_q = best_ohp_q - base_q
    pct_q = (diff_q / base_q) * 100.0 if base_q > 0 else 0.0
    winner_q = 'OHP-MOCD' if best_ohp_q > base_q else 'Çetin 2022'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'Çetin 2022 (Shen Q)',
        'Dataset': n_str,
        'Metric': 'Shen Q (EQ)',
        'Algorithm X Reported Score': f"{base_q:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{b_q:.4f}",
        'OHP-MOCD (Crisp)': f"{c_q:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp_q:.4f}",
        'Absolute Diff (Delta)': f"{diff_q:+.4f}",
        'Pct Improvement (%)': f"{pct_q:+.2f}%",
        'Outperforming Algorithm': winner_q
    })
    
    # Coverage
    b_cov = r['OHP_MOCD_BoundarySeeded_Coverage']
    c_cov = r['OHP_MOCD_Crisp_Coverage']
    best_ohp_cov = max(b_cov, c_cov)
    base_cov = r['Proposed_Cetin_Coverage']
    diff_cov = best_ohp_cov - base_cov
    pct_cov = (diff_cov / base_cov) * 100.0 if base_cov > 0 else 0.0
    winner_cov = 'OHP-MOCD' if best_ohp_cov > base_cov else 'Çetin 2022'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'Çetin 2022 (Coverage)',
        'Dataset': n_str,
        'Metric': 'Coverage (Formula 9)',
        'Algorithm X Reported Score': f"{base_cov:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{b_cov:.4f}",
        'OHP-MOCD (Crisp)': f"{c_cov:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp_cov:.4f}",
        'Absolute Diff (Delta)': f"{diff_cov:+.4f}",
        'Pct Improvement (%)': f"{pct_cov:+.2f}%",
        'Outperforming Algorithm': winner_cov
    })

df_master = pd.DataFrame(master_rows)
output_path = BENCH_DIR / "master_unified_comparative_table.csv"
df_master.to_csv(output_path, index=False)
print(f"Saved Master Unified Table CSV to: {output_path}")

print("\n" + df_master.to_string(index=False))
