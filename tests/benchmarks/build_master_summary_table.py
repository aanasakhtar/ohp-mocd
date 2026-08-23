"""
build_master_summary_table.py

Builds the Master Unified Comparative Table across all 5 Literature-Reported Baseline Papers:
  1. SLPA (Xie & Szymanski, IEEE TKDE 2011/2012)
  2. MCMOEA (Wen et al., IEEE TEVC 2016)
  3. Çetin & Amrahov (Kybernetika 2022)
  4. LPAM (Ponomarenko et al., PLOS ONE 2021)
  5. NOCD (Shchur & Günnemann, KDD / ICLR 2019)
"""

import os
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

master_rows = []

# 1. Algorithm X = SLPA (2011)
f1 = BENCH_DIR / "strict_paper1_slpa_qov.csv"
if f1.exists():
    df1 = pd.read_csv(f1)
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
f2 = BENCH_DIR / "strict_paper2_mcmoea_qov.csv"
if f2.exists():
    df2 = pd.read_csv(f2)
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

# 3. Algorithm X = Çetin & Amrahov (2022)
f3 = BENCH_DIR / "strict_paper3_cetin_q_coverage.csv"
if f3.exists():
    df3 = pd.read_csv(f3)
    for _, r in df3.iterrows():
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

# 4. Algorithm X = LPAM (2021)
f4 = BENCH_DIR / "strict_paper4_lpam_onmi_f1.csv"
if f4.exists():
    df4 = pd.read_csv(f4)
    for _, r in df4.iterrows():
        net = r['Dataset']
        # ONMI
        b_onmi = r['OHP_MOCD_BoundarySeeded_ONMI_Peak']
        c_onmi = r['OHP_MOCD_Crisp_ONMI']
        best_ohp_onmi = max(b_onmi, c_onmi)
        base_onmi = r['LPAM_ONMI_Reported']
        diff_onmi = best_ohp_onmi - base_onmi
        pct_onmi = (diff_onmi / base_onmi) * 100.0 if base_onmi > 0 else 0.0
        winner_onmi = 'OHP-MOCD' if best_ohp_onmi >= base_onmi else 'LPAM'
        master_rows.append({
            'Baseline Paper Algorithm (X)': 'LPAM 2021 (ONMI)',
            'Dataset': net,
            'Metric': 'ONMI (LFK)',
            'Algorithm X Reported Score': f"{base_onmi:.4f}",
            'OHP-MOCD (BoundarySeeded)': f"{b_onmi:.4f}",
            'OHP-MOCD (Crisp)': f"{c_onmi:.4f}",
            'Best OHP-MOCD Score': f"{best_ohp_onmi:.4f}",
            'Absolute Diff (Delta)': f"{diff_onmi:+.4f}",
            'Pct Improvement (%)': f"{pct_onmi:+.2f}%",
            'Outperforming Algorithm': winner_onmi
        })
        
        # F1
        b_f1 = r['OHP_MOCD_BoundarySeeded_F1_Peak']
        c_f1 = r['OHP_MOCD_Crisp_F1']
        best_ohp_f1 = max(b_f1, c_f1)
        base_f1 = r['LPAM_F1_Reported']
        diff_f1 = best_ohp_f1 - base_f1
        pct_f1 = (diff_f1 / base_f1) * 100.0 if base_f1 > 0 else 0.0
        winner_f1 = 'OHP-MOCD' if best_ohp_f1 >= base_f1 else 'LPAM'
        master_rows.append({
            'Baseline Paper Algorithm (X)': 'LPAM 2021 (F1)',
            'Dataset': net,
            'Metric': 'Pairwise F1',
            'Algorithm X Reported Score': f"{base_f1:.4f}",
            'OHP-MOCD (BoundarySeeded)': f"{b_f1:.4f}",
            'OHP-MOCD (Crisp)': f"{c_f1:.4f}",
            'Best OHP-MOCD Score': f"{best_ohp_f1:.4f}",
            'Absolute Diff (Delta)': f"{diff_f1:+.4f}",
            'Pct Improvement (%)': f"{pct_f1:+.2f}%",
            'Outperforming Algorithm': winner_f1
        })

# 5. Algorithm X = NOCD (2019)
f5 = BENCH_DIR / "strict_paper5_nocd_onmi.csv"
if f5.exists():
    df5 = pd.read_csv(f5)
    for _, r in df5.iterrows():
        net = r['Dataset']
        b_onmi = r['OHP_MOCD_BoundarySeeded_ONMI_Peak']
        c_onmi = r['OHP_MOCD_Crisp_ONMI']
        best_ohp_onmi = max(b_onmi, c_onmi)
        base_onmi = r['NOCD_G_ONMI_Reported']
        diff_onmi = best_ohp_onmi - base_onmi
        pct_onmi = (diff_onmi / base_onmi) * 100.0 if base_onmi > 0 else 0.0
        winner_onmi = 'OHP-MOCD' if best_ohp_onmi >= base_onmi else 'NOCD'
        master_rows.append({
            'Baseline Paper Algorithm (X)': 'NOCD 2019 (ONMI)',
            'Dataset': f"{net} (N = {r['Nodes']})",
            'Metric': 'ONMI (McDaid)',
            'Algorithm X Reported Score': f"{base_onmi:.4f}",
            'OHP-MOCD (BoundarySeeded)': f"{b_onmi:.4f}",
            'OHP-MOCD (Crisp)': f"{c_onmi:.4f}",
            'Best OHP-MOCD Score': f"{best_ohp_onmi:.4f}",
            'Absolute Diff (Delta)': f"{diff_onmi:+.4f}",
            'Pct Improvement (%)': f"{pct_onmi:+.2f}%",
            'Outperforming Algorithm': winner_onmi
        })

df_master = pd.DataFrame(master_rows)
output_path = BENCH_DIR / "master_unified_comparative_table.csv"
df_master.to_csv(output_path, index=False)
print(f"Saved Master Unified Table CSV to: {output_path}")

print("\n" + df_master.to_string(index=False))
