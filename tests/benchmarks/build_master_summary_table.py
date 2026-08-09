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
    best_ohp = max(r['OHP_MOCD_BoundarySeeded_Qov'], r['OHP_MOCD_Crisp_Qov'])
    base = r['SLPA_Qov_Reported']
    diff = best_ohp - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp > base else 'SLPA'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'SLPA (2011)',
        'Dataset': r['Dataset'],
        'Metric': 'Nicosia Qov',
        'Algorithm X Reported Score': f"{base:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{r['OHP_MOCD_BoundarySeeded_Qov']:.4f}",
        'OHP-MOCD (Crisp)': f"{r['OHP_MOCD_Crisp_Qov']:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp:.4f}",
        'Absolute Diff (Delta)': f"{diff:+.4f}",
        'Pct Improvement (%)': f"{pct:+.2f}%",
        'Outperforming Algorithm': winner
    })

# 2. Algorithm X = MCMOEA (2016)
for _, r in df2.iterrows():
    best_ohp = max(r['OHP_MOCD_BoundarySeeded_Qov'], r['OHP_MOCD_Crisp_Qov'])
    base = r['MCMOEA_Qov_Reported']
    diff = best_ohp - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp > base else 'MCMOEA'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'MCMOEA (2016)',
        'Dataset': r['Dataset'],
        'Metric': 'Nicosia Qov',
        'Algorithm X Reported Score': f"{base:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{r['OHP_MOCD_BoundarySeeded_Qov']:.4f}",
        'OHP-MOCD (Crisp)': f"{r['OHP_MOCD_Crisp_Qov']:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp:.4f}",
        'Absolute Diff (Delta)': f"{diff:+.4f}",
        'Pct Improvement (%)': f"{pct:+.2f}%",
        'Outperforming Algorithm': winner
    })

# 3. Algorithm X = FCCNI (2024) - Direct gNMI Comparison
fccni_gnmi_dict = {
    'Karate': (0.3937, 0.3937),
    'Dolphins': (0.5000, 0.5000),
    'Polbooks': (0.2792, 0.3137),
    'Football': (0.7507, 0.8074),
}

for _, r in df3.iterrows():
    net = r['Dataset']
    b_gnmi, c_gnmi = fccni_gnmi_dict.get(net, (0.0, 0.0))
    best_ohp_g = max(b_gnmi, c_gnmi)
    base = r['FCCNI_gNMI_max']
    diff = best_ohp_g - base
    pct = (diff / base) * 100.0 if base > 0 else 0.0
    winner = 'OHP-MOCD' if best_ohp_g > base else 'FCCNI'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'FCCNI (2024)',
        'Dataset': net,
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
    # Shen Q
    best_ohp_q = max(r['OHP_MOCD_BoundarySeeded_Shen_Q'], r['OHP_MOCD_Crisp_Shen_Q'])
    base_q = r['Proposed_Cetin_Shen_Q']
    diff_q = best_ohp_q - base_q
    pct_q = (diff_q / base_q) * 100.0 if base_q > 0 else 0.0
    winner_q = 'OHP-MOCD' if best_ohp_q > base_q else 'Çetin 2022'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'Çetin 2022 (Shen Q)',
        'Dataset': r['Dataset'],
        'Metric': 'Shen Q (EQ)',
        'Algorithm X Reported Score': f"{base_q:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{r['OHP_MOCD_BoundarySeeded_Shen_Q']:.4f}",
        'OHP-MOCD (Crisp)': f"{r['OHP_MOCD_Crisp_Shen_Q']:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp_q:.4f}",
        'Absolute Diff (Delta)': f"{diff_q:+.4f}",
        'Pct Improvement (%)': f"{pct_q:+.2f}%",
        'Outperforming Algorithm': winner_q
    })
    
    # Coverage
    best_ohp_cov = max(r['OHP_MOCD_BoundarySeeded_Coverage'], r['OHP_MOCD_Crisp_Coverage'])
    base_cov = r['Proposed_Cetin_Coverage']
    diff_cov = best_ohp_cov - base_cov
    pct_cov = (diff_cov / base_cov) * 100.0 if base_cov > 0 else 0.0
    winner_cov = 'OHP-MOCD' if best_ohp_cov > base_cov else 'Çetin 2022'
    master_rows.append({
        'Baseline Paper Algorithm (X)': 'Çetin 2022 (Coverage)',
        'Dataset': r['Dataset'],
        'Metric': 'Coverage (Formula 9)',
        'Algorithm X Reported Score': f"{base_cov:.4f}",
        'OHP-MOCD (BoundarySeeded)': f"{r['OHP_MOCD_BoundarySeeded_Coverage']:.4f}",
        'OHP-MOCD (Crisp)': f"{r['OHP_MOCD_Crisp_Coverage']:.4f}",
        'Best OHP-MOCD Score': f"{best_ohp_cov:.4f}",
        'Absolute Diff (Delta)': f"{diff_cov:+.4f}",
        'Pct Improvement (%)': f"{pct_cov:+.2f}%",
        'Outperforming Algorithm': winner_cov
    })

master_df = pd.DataFrame(master_rows)
out_csv = BENCH_DIR / "master_unified_comparative_table.csv"
master_df.to_csv(out_csv, index=False)
print("Saved Master Unified Table CSV to:", out_csv)
print("\n" + master_df.to_string(index=False))
