import sys, os, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "v3_main"

def load(R):
    G = {}
    G["d"] = pd.read_csv(f"{R}/table_discrimination_calibration.csv")
    G["nb"] = pd.read_csv(f"{R}/table_net_benefit.csv")
    G["sg"] = pd.read_csv(f"{R}/table_subgroups.csv")
    G["fr"] = pd.read_csv(f"{R}/table_frs_benchmark.csv")
    G["fb"] = pd.read_csv(f"{R}/table_fullboot_noninferiority.csv")
    G["sh"] = pd.read_csv(f"{R}/shap_mean_abs.csv")
    G["t1"] = pd.read_csv(f"{R}/table1_characteristics.csv")
    G["att"] = pd.read_csv(f"{R}/attrition_exam3.csv")
    return G

SET = ["A_Cleveland", "A_Hungary", "A_VA", "B1_UCI_to_Fram", "C0_Fram_disjoint", "C_Fram_temporal"]
ML = ["RF", "GB", "SVM", "MLP", "XGB"]
SIX = ["LR"] + ML

if __name__ == "__main__":
    G = load(R); d = G["d"]
    m = d[d.scenario.isin(SET) & d.model.isin(ML)]
    print("paired ML vs LR comparisons:", len(m), "better(lo>0):", int((m.dauc_vs_LR_lo > 0).sum()),
          "worse(hi<0):", int((m.dauc_vs_LR_hi < 0).sum()), "point<0:", int((m.dauc_vs_LR < 0).sum()), "point>0:", int((m.dauc_vs_LR > 0).sum()))
    print(m[m.dauc_vs_LR_hi < 0][["scenario", "model", "dauc_vs_LR", "dauc_vs_LR_lo", "dauc_vs_LR_hi"]].round(3))
    for s in SET:
        x = d[(d.scenario == s) & d.model.isin(SIX)]
        print(s, "AUC range", x.auc.min().round(3), x.auc.max().round(3), "best ext", x.loc[x.auc.idxmax(), "model"],
              "best int", x.loc[x.internal_cv_auc.idxmax(), "model"], "LR drop", x[x.model == "LR"].auc_drop.iloc[0].round(3),
              "mean drop", x.auc_drop.mean().round(3), "drop range", x.auc_drop.min().round(3), x.auc_drop.max().round(3))
        a = d[(d.scenario == s) & (d.model == "LR")]
        print("   LR vs AGE", a.dauc_vs_AGE.iloc[0].round(3), a.dauc_vs_AGE_lo.iloc[0].round(3), a.dauc_vs_AGE_hi.iloc[0].round(3))
    ag = d[d.scenario.isin(SET) & d.model.isin(SIX)]
    print("models beating AGE with CI excl 0:", int((ag.dauc_vs_AGE_lo > 0).sum()), "of", len(ag))
    fb = G["fb"]; f = fb[(fb.ref == "LR") & fb.model.isin(ML)]
    for sc in f.scenario.unique():
        z = f[f.scenario == sc]
        print(sc, {k: int(z[k].sum()) for k in ["LR_noninf_2", "LR_noninf_3", "LR_noninf_5", "ML_noninf_2", "ML_noninf_3", "ML_noninf_5"]}, "n", len(z),
              "any ML sig better", int((z.d_lo > 0).sum()))
    sg = G["sg"]; x = sg[(sg.kind == "diff") & sg.scenario.isin(SET) & sg.model.isin(SIX)]
    for k in ["dsens_sex", "dspec_sex", "dsens_age", "dspec_age"]:
        z = x[(x.group == k) & x.est.notna()]
        print(k, "n eval", len(z), "excl0", int(((z.lo > 0) | (z.hi < 0)).sum()), "range", z.est.min().round(2), z.est.max().round(2))
