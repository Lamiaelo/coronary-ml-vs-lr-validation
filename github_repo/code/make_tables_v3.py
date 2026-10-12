import sys, json, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = sys.argv[1]; O = sys.argv[2]
d = pd.read_csv(f"{R}/table_discrimination_calibration.csv")
nb = pd.read_csv(f"{R}/table_net_benefit.csv")
dca = pd.read_csv(f"{R}/dca_curves.csv")
cb = pd.read_csv(f"{R}/calibration_bins.csv")
sg = pd.read_csv(f"{R}/table_subgroups.csv")
fr = pd.read_csv(f"{R}/table_frs_benchmark.csv")
try: fb = pd.read_csv(f"{R}/table_fullboot_noninferiority.csv")
except Exception: fb = None
MAIN = ["A_Cleveland","A_Hungary","A_VA","B1_UCI_to_Fram","B2_Fram_to_UCI","C0_Fram_disjoint","C_Fram_temporal"]
MOD = ["LR","RF","GB","SVM","MLP","XGB","AGE","LRX"]
f2 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
def ci(r, a, lo, hi): return f"{f2(r[a])} ({f2(r[lo])}–{f2(r[hi])})"
T = {}
# discrimination table
rows = []
for s in MAIN:
    for m in MOD:
        r = d[(d.scenario == s) & (d.model == m)]
        if len(r):
            r = r.iloc[0]
            rows.append([s, m, ci(r,"auc","auc_lo","auc_hi"),
                         "" if m=="LR" else ci(r,"dauc_vs_LR","dauc_vs_LR_lo","dauc_vs_LR_hi"),
                         ci(r,"slope","slope_lo","slope_hi"), ci(r,"citl","citl_lo","citl_hi"),
                         f2(r["brier"]), f2(r["brier_r2"]), f2(r["internal_cv_auc"])])
T["disc"] = rows
d.to_json(f"{O}/disc.json", orient="records")
json.dump(T, open(f"{O}/tables_v3.json","w"))
# figures
col = dict(zip(MOD, ["#1f4e79","#2e8b57","#c0762b","#7b3f98","#b22222","#5f6b73","#999999","#000000"]))
# Fig AUC forest
fig, axs = plt.subplots(2, 3, figsize=(11, 6.4), sharey=True)
for ax, s in zip(axs.ravel(), MAIN[:3] + MAIN[3:4] + MAIN[5:7]):
    for i, m in enumerate(MOD):
        r = d[(d.scenario==s)&(d.model==m)]
        if not len(r): continue
        r = r.iloc[0]
        ax.errorbar(r.auc, i, xerr=[[r.auc-r.auc_lo],[r.auc_hi-r.auc]], fmt="o", color=col[m], capsize=3, ms=5)
    ax.set_title({"A_Cleveland":"Cleveland (held out)","A_Hungary":"Hungary (held out)","A_VA":"VA (held out)","B1_UCI_to_Fram":"UCI to Framingham (B1)","C0_Fram_disjoint":"Framingham, same exam (C0)","C_Fram_temporal":"Framingham, exam 3 (C)"}[s], fontsize=10)
    ax.axvline(0.5, color="#ccc", lw=.8); ax.set_xlim(0.45,0.85); ax.tick_params(labelsize=9); ax.grid(axis="x", color="#eee")
    ax.set_yticks(range(len(MOD))); ax.set_yticklabels(["LR","RF","GB","SVM","MLP","XGBoost","Age only","Extended LR"], fontsize=9)
axs[0,0].invert_yaxis()
fig.supxlabel("AUC (95% bootstrap interval)", fontsize=10); plt.tight_layout(); plt.savefig(f"{O}/fig_auc.png", dpi=200); plt.close()
# Fig calibration
sc = ["A_Cleveland","B1_UCI_to_Fram","C_Fram_temporal"]
fig, axs = plt.subplots(1,3, figsize=(11,3.8))
for ax, s in zip(axs, sc):
    ax.plot([0,1],[0,1], color="#aaa", lw=.8)
    for m in ["LR","RF","GB","SVM","MLP","XGB"]:
        b = cb[(cb.scenario==s)&(cb.model==m)]
        ax.plot(b.mean_pred, b.obs_rate, "-o", ms=3, lw=1, color=col[m], label=m)
    ax.set_title(s.replace("_"," "), fontsize=9); ax.set_xlabel("Mean predicted", fontsize=8); ax.set_ylabel("Observed", fontsize=8)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{O}/fig_calib.png", dpi=200); plt.close()
# Fig DCA (recalibrated)
sc = ["A_Cleveland","A_VA","C_Fram_temporal"]
fig, axs = plt.subplots(1,3, figsize=(11,3.8))
for ax, s in zip(axs, sc):
    for m in ["LR","GB","MLP","AGE"]:
        b = dca[(dca.scenario==s)&(dca.model==m)&(dca.kind=="recal")]
        if not len(b): b = dca[(dca.scenario==s)&(dca.model==m)]
        b = b.sort_values("t"); ax.plot(b.t, b.nb, color=col[m], label=m, lw=1.2)
    p = d[(d.scenario==s)&(d.model=="LR")].prevalence.iloc[0]
    t = np.linspace(0.05,0.8,50); ax.plot(t, p-(1-p)*t/(1-t), "--", color="#777", lw=.8, label="Treat all")
    ax.axhline(0, color="#444", lw=.6); ax.set_ylim(-0.05, p+0.02); ax.set_title(s.replace("_"," "), fontsize=9)
    ax.set_xlabel("Threshold probability", fontsize=8); ax.set_ylabel("Net benefit", fontsize=8)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{O}/fig_dca.png", dpi=200); plt.close()
print("ok")
# ---- design figure
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, t, fc="#e8eef5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#34495e", lw=1))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.2))
ax.text(1, 57, "Development data", fontsize=9, weight="bold"); ax.text(66, 57, "Test data", fontsize=9, weight="bold"); ax.text(41, 57, "Setting", fontsize=9, weight="bold")
rows = [("UCI: two sites", "A  leave-one-site-out", "UCI: third site\n(angiographic disease)", "#e8eef5"),
        ("UCI: three sites pooled", "B1  cross-outcome", "Framingham exam 1\n(10-year CHD)", "#f7ebe0"),
        ("Framingham exam 1\n(all participants)", "B2  cross-outcome", "UCI: each site", "#f7ebe0"),
        ("Framingham exam 1\nhalf 1 of participants", "C0  same exam,\nnew participants", "Framingham exam 1\nhalf 2 of participants", "#e6f2e6"),
        ("Framingham exam 1\nhalf 1 of participants", "C  later exam,\nnew participants", "Framingham exam 3\nhalf 2 of participants", "#e6f2e6")]
for i, (a, b, c, col) in enumerate(rows):
    y = 46 - i * 10.5
    box(1, y, 30, 8, a, col); box(37, y, 24, 8, b, "#ffffff"); box(66, y, 31, 8, c, col)
    arrow(31.4, y + 4, 36.6, y + 4); arrow(61.4, y + 4, 65.6, y + 4)
ax.text(1, 1.5, "Predictors: age, sex, cholesterol, systolic BP, hyperglycaemia. Imputation, tuning, thresholds fitted on development data only.", fontsize=7.5, style="italic")
plt.savefig(f"{O}/fig1_design_v3.png", dpi=200, bbox_inches="tight"); plt.close()
