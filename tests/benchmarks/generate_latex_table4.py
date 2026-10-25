import pandas as pd

df = pd.read_csv("tests/benchmarks/abcd_synthetic_benchmark_master_results.csv")
name_order = [
    "EF-MOCD (2020)",
    "LPAM (2021)",
    "MCMOEA (2016)",
    "MO-EE (2018)",
    "NOCD (2019)",
    "OHP-MOCD (Proposed)",
    "SLPA (2011)"
]

lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\fontsize{6.2pt}{7.2pt}\\selectfont")
lines.append("\\renewcommand{\\arraystretch}{0.48}")
lines.append("\\setlength{\\tabcolsep}{2.0pt}")
lines.append("\\caption{Comparative performance on modern synthetic $\\text{ABCD}+o^2$ benchmark networks across background mixing parameter values $\\xi \\in [0.1, 0.6]$ with overlap multiplier $\\eta = 1.50$ (mean $\\pm$ std across 5 independent seeds; runtime reports mean wall-clock execution time). Boldface indicates the best mean value among all compared methods.}")
lines.append("\\label{tab:abcd_results}")
lines.append("\\resizebox{0.94\\textwidth}{!}{%")
lines.append("\\begin{tabular}{@{}clcccc@{}}")
lines.append("\\toprule")
lines.append("\\textbf{Mixing Parameter ($\\xi$)} & \\textbf{Algorithm} & \\textbf{ONMI} & \\textbf{Pairwise $F_1$} & \\textbf{Modularity ($EQ$)} & \\textbf{Runtime (s)} \\\\")
lines.append("\\midrule")

for xi_idx, xi in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    sub = df[df["xi"] == xi].copy()
    sub["sort_idx"] = sub["Algorithm"].map(lambda x: name_order.index(x) if x in name_order else 99)
    sub = sub.sort_values("sort_idx")
    max_onmi = sub["ONMI_mean"].max()
    max_f1 = sub["F1_mean"].max()
    max_eq = sub["EQ_mean"].max()
    min_time = sub["Runtime_mean"].min()
    
    first = True
    for _, r in sub.iterrows():
        xi_str = f"$\\xi = {xi:.1f}$" if first else ""
        first = False
        algo = r["Algorithm"]
        algo_tex = f"\\textbf{{{algo}}}" if "OHP-MOCD" in algo else algo
        
        onmi_val = f"{r['ONMI_mean']:.4f} \\pm {r['ONMI_std']:.4f}"
        if abs(r["ONMI_mean"] - max_onmi) < 1e-6:
            onmi_val = f"\\mathbf{{{onmi_val}}}"
            
        f1_val = f"{r['F1_mean']:.4f} \\pm {r['F1_std']:.4f}"
        if abs(r["F1_mean"] - max_f1) < 1e-6:
            f1_val = f"\\mathbf{{{f1_val}}}"
            
        eq_val = f"{r['EQ_mean']:.4f} \\pm {r['EQ_std']:.4f}"
        if abs(r["EQ_mean"] - max_eq) < 1e-6:
            eq_val = f"\\mathbf{{{eq_val}}}"
            
        time_val = f"{r['Runtime_mean']:.2f}\\text{{s}}"
        if abs(r["Runtime_mean"] - min_time) < 1e-6:
            time_val = f"\\mathbf{{{time_val}}}"
            
        lines.append(f" {xi_str} & {algo_tex} & ${onmi_val}$ & ${f1_val}$ & ${eq_val}$ & {time_val} \\\\")
        
    if xi_idx < 5:
        lines.append("\\midrule")
    else:
        lines.append("\\bottomrule")

lines.append("\\end{tabular}%")
lines.append("}")
lines.append("\\end{table}")

output_tex = "\n".join(lines)
with open("tests/benchmarks/table4_abcd_generated.tex", "w", encoding="utf-8") as f:
    f.write(output_tex)
print(output_tex)
