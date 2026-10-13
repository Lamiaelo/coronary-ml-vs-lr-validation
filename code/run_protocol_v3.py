#!/usr/bin/env python3

import argparse, itertools, json, os, pickle, platform, sys, time, warnings, zlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
FEATS = ["age", "sex", "chol", "sbp", "hyperglycaemia"]
EXT = FEATS + ["cursmoke", "cigs", "bpmeds", "diabetes", "bmi"]
MODELS = ["LR", "RF", "GB", "SVM", "MLP", "XGB"]
BASE, EXTM = "AGE", "LRX"
FMAP = {**{m: FEATS for m in MODELS}, BASE: FEATS, EXTM: EXT}
TEN = 3652.5
FAST = False
DCA_T = {"uci": [0.30, 0.50], "fram": [0.10, 0.20]}
DCA_GRID = {"uci": np.round(np.arange(0.05, 0.951, 0.05), 2), "fram": np.round(np.arange(0.01, 0.401, 0.01), 2)}
MIN_EVENTS = 20


# ------------------------------------------------------------------------------ data
UCI_FILES = {"Cleveland": "processed.cleveland.data", "Hungary": "processed.hungarian.data",
             "Switzerland": "processed.switzerland.data", "VA": "processed.va.data"}


def load_uci(uci_dir, include_swiss=False):
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "num"]
    out = {}
    for site, fn in UCI_FILES.items():
        if site == "Switzerland" and not include_swiss:
            continue
        d = pd.read_csv(os.path.join(uci_dir, fn), header=None, names=cols, na_values="?").apply(pd.to_numeric, errors="coerce")
        d["chol"] = d["chol"].replace(0, np.nan)          # 0 = missing-value code
        d["trestbps"] = d["trestbps"].replace(0, np.nan)
        df = pd.DataFrame({"age": d.age, "sex": d.sex, "chol": d.chol, "sbp": d.trestbps,
                           "hyperglycaemia": d.fbs, "y": (d.num > 0).astype(int)})
        for c in ["cursmoke", "cigs", "bpmeds", "diabetes", "bmi", "hdl"]:
            df[c] = np.nan
        df["id"] = np.arange(len(df))
        out[site] = df
    return out


def load_fram(path, period, outcome="chd10"):
    d = pd.read_csv(path)
    d = d[(d.PERIOD == period) & (d.PREVCHD == 0)].copy()
    rel, drel = d.TIMECHD - d.TIME, d.TIMEDTH - d.TIME
    chd = ((d.ANYCHD == 1) & (rel > 0) & (rel <= TEN)).astype(int)
    dth = ((d.DEATH == 1) & (drel > 0) & (drel <= TEN)).astype(int)
    keep = pd.Series(True, index=d.index)
    if outcome == "chd10":
        y = chd
    elif outcome == "composite":
        y = ((chd == 1) | (dth == 1)).astype(int)
    elif outcome == "excl_noncvd_death":
        y = chd; keep = ~((dth == 1) & (chd == 0))
    else:
        raise ValueError(outcome)
    hyper = (d.GLUCOSE > 120).astype(float)
    hyper[d.GLUCOSE.isna()] = np.nan
    hyper[d.DIABETES == 1] = 1.0
    out = pd.DataFrame({"age": d.AGE, "sex": (d.SEX == 1).astype(int), "chol": d.TOTCHOL, "sbp": d.SYSBP,
                        "hyperglycaemia": hyper, "cursmoke": d.CURSMOKE, "cigs": d.CIGPDAY, "bpmeds": d.BPMEDS,
                        "diabetes": d.DIABETES, "bmi": d.BMI, "hdl": d.HDLC, "y": y, "id": d.RANDID,
                        "death10": dth})
    return out[keep].reset_index(drop=True)


def frs_cvd(df):
    """Published 10-year general CVD risk equation (D'Agostino et al., Circulation 2008; Cox model with
    log-transformed predictors). Used ONLY as an untuned clinical benchmark for discrimination; its outcome
    (general CVD) differs from the CHD outcome analysed here, so calibration is not meaningful."""
    ln = np.log
    male = df.sex.values == 1
    treated = df.bpmeds.values == 1
    smoke, dm = df.cursmoke.values, df.diabetes.values
    lp_m = (3.06117 * ln(df.age) + 1.12370 * ln(df.chol) - 0.93263 * ln(df.hdl)
            + np.where(treated, 1.99881, 1.93303) * ln(df.sbp) + 0.65451 * smoke + 0.57367 * dm)
    lp_f = (2.32888 * ln(df.age) + 1.20904 * ln(df.chol) - 0.70833 * ln(df.hdl)
            + np.where(treated, 2.82263, 2.76157) * ln(df.sbp) + 0.52873 * smoke + 0.69154 * dm)
    return np.where(male, 1 - 0.88936 ** np.exp(lp_m - 23.9802), 1 - 0.95012 ** np.exp(lp_f - 26.1931))


# ------------------------------------------------------------------------------ models
def pipe(clf, cols=None):
    steps = []
    if cols is not None:
        steps.append(("sel", ColumnTransformer([("c", "passthrough", cols)])))
    steps += [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)]
    return Pipeline(steps)


def model_grid(seed):
    g = {
        "LR": (pipe(LogisticRegression(max_iter=2000)), {"clf__C": [0.01, 0.1, 1, 10]}),
        "RF": (pipe(RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)),
               {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [5, 15]}),
        "GB": (pipe(GradientBoostingClassifier(random_state=seed)),
               {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]}),
        # unweighted RBF-SVM collapses at low prevalence; balanced weights + Platt scaling
        "SVM": (pipe(CalibratedClassifierCV(SVC(class_weight="balanced", random_state=seed), method="sigmoid", cv=3)),
                {"clf__estimator__C": [0.01, 0.1, 1]}),
        # default early stopping monitors accuracy and stalls at low prevalence; use L2 regularisation instead
        "MLP": (pipe(MLPClassifier(max_iter=1000, random_state=seed, early_stopping=False, learning_rate_init=0.005)),
                {"clf__hidden_layer_sizes": [(8,), (16, 8)], "clf__alpha": [1e-2, 1e-1, 1.0]}),
        "XGB": (pipe(XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0)),
                {"clf__max_depth": [2, 3], "clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]}),
        BASE: (pipe(LogisticRegression(max_iter=2000), cols=["age"]), {"clf__C": [1.0]}),
        EXTM: (pipe(LogisticRegression(max_iter=2000)), {"clf__C": [0.01, 0.1, 1, 10]}),
    }
    if FAST:
        g = {k: (e, {kk: vv[:1] for kk, vv in gr.items()}) for k, (e, gr) in g.items()}
    return g


def fit_models(train, seed, use_ext):
    names = MODELS + [BASE] + ([EXTM] if use_ext else [])
    grid = model_grid(seed); y = train["y"].values
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    fitted, oof, bp = {}, {}, {}
    for m in names:
        est, g = grid[m]
        gs = GridSearchCV(est, g, scoring="roc_auc", cv=inner, n_jobs=2).fit(train[FMAP[m]], y)
        fitted[m] = gs.best_estimator_; bp[m] = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
        oof[m] = cross_val_predict(gs.best_estimator_, train[FMAP[m]], y, cv=inner, method="predict_proba")[:, 1]
    return fitted, oof, bp


# ------------------------------------------------------------------------------ metrics
def irls(X, y, off=None, iters=50):
    beta = np.zeros(X.shape[1]); off = np.zeros(len(y)) if off is None else off
    for _ in range(iters):
        eta = X @ beta + off; mu = expit(eta); w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta - off + (y - mu) / w
        XtW = X.T * w
        nb = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        done = np.max(np.abs(nb - beta)) < 1e-8
        beta = nb
        if done: break
    return beta


def calib_fast(y, p):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    try: slope = irls(np.c_[np.ones(len(y)), lp], y.astype(float))[1]
    except Exception: slope = np.nan
    try: citl = irls(np.ones((len(y), 1)), y.astype(float), off=lp)[0]
    except Exception: citl = np.nan
    return float(slope), float(citl)


def crossfit_recal(y, p, mode, seed, k=5):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6)); out = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(lp, y):
        if mode == "slope":
            b = irls(np.c_[np.ones(len(tr)), lp[tr]], y[tr].astype(float)); out[te] = expit(b[0] + b[1] * lp[te])
        else:
            a = irls(np.ones((len(tr), 1)), y[tr].astype(float), off=lp[tr])[0]; out[te] = expit(a + lp[te])
    return out


def youden(y, p):
    fpr, tpr_, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr_ - fpr)])


def net_benefit(y, p, t):
    pred = p >= t
    return (np.sum(pred & (y == 1)) - np.sum(pred & (y == 0)) * t / (1 - t)) / len(y)


def nb_all(y, t):
    prev = y.mean()
    return prev - (1 - prev) * t / (1 - t)


def strat_idx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def pct(a):
    a = np.asarray(a, float)
    return tuple(np.nanpercentile(a, [2.5, 97.5])) if np.isfinite(a).sum() > 20 else (np.nan, np.nan)


def sens_spec(yb, pr):
    pos, neg = yb == 1, yb == 0
    se = pr[pos].mean() if pos.sum() >= MIN_EVENTS else np.nan
    sp = (~pr[neg]).mean() if neg.sum() >= MIN_EVENTS else np.nan
    return se, sp


def sub_diffs(yb, pr, sb, ab, cut):
    out = {}
    for nm, ga, gb in [("sex", sb == 1, sb == 0), ("age", ab < cut, ab >= cut)]:
        sa, pa = sens_spec(yb[ga], pr[ga]); sb_, pb_ = sens_spec(yb[gb], pr[gb])
        out[f"dsens_{nm}"] = sa - sb_ if np.isfinite(sa) and np.isfinite(sb_) else np.nan
        out[f"dspec_{nm}"] = pa - pb_ if np.isfinite(pa) and np.isfinite(pb_) else np.nan
    return out


# ------------------------------------------------------------------------------ evaluation of one held-out set
def evaluate(scn, site, test, preds, thr, age_cut, family, B, rng, seed):
    y, sex, age = test["y"].values, test["sex"].values, test["age"].values
    names = list(preds); tgrid = DCA_GRID[family]; tpre = DCA_T[family]
    r1 = {m: crossfit_recal(y, preds[m], "int", seed) for m in names}
    r2 = {m: crossfit_recal(y, preds[m], "slope", seed) for m in names}
    kinds = {"raw": preds, "recal": r2}
    S = {k: {m: [] for m in names} for k in ["auc", "brier", "slope", "citl", "brier_r1", "brier_r2"]}
    D = {k: {m: [] for m in names} for k in ["dsens_sex", "dspec_sex", "dsens_age", "dspec_age"]}
    NB = {(m, kd, t): [] for m in names for kd in kinds for t in tpre}
    NA = {t: [] for t in tpre}
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y)); yb = y[ix]   # ordinary (non-stratified) bootstrap of patients: keeps event-rate variability
        if yb.min() == yb.max(): continue
        for t in tpre: NA[t].append(nb_all(yb, t))
        for m in names:
            pb = preds[m][ix]
            S["auc"][m].append(roc_auc_score(yb, pb)); S["brier"][m].append(brier_score_loss(yb, pb))
            sl, ci = calib_fast(yb, pb); S["slope"][m].append(sl); S["citl"][m].append(ci)
            S["brier_r1"][m].append(brier_score_loss(yb, r1[m][ix])); S["brier_r2"][m].append(brier_score_loss(yb, r2[m][ix]))
            dd = sub_diffs(yb, pb >= thr[m], sex[ix], age[ix], age_cut)
            for k in D: D[k][m].append(dd[k])
            for kd, P in kinds.items():
                for t in tpre: NB[(m, kd, t)].append(net_benefit(yb, P[m][ix], t))
    # point estimates
    rows, nbrows, subrows, curves, bins = [], [], [], [], []
    ref_auc = {r: roc_auc_score(y, preds[r]) for r in ["LR", BASE]}
    for m in names:
        auc = roc_auc_score(y, preds[m]); sl, ci = calib_fast(y, preds[m]); sl2, ci2 = calib_fast(y, r2[m])
        r = dict(scenario=scn, test=site, model=m, n=len(y), events=int(y.sum()), prevalence=y.mean(), auc=auc,
                 auc_lo=pct(S["auc"][m])[0], auc_hi=pct(S["auc"][m])[1], brier=brier_score_loss(y, preds[m]),
                 brier_lo=pct(S["brier"][m])[0], brier_hi=pct(S["brier"][m])[1], slope=sl, slope_lo=pct(S["slope"][m])[0],
                 slope_hi=pct(S["slope"][m])[1], citl=ci, citl_lo=pct(S["citl"][m])[0], citl_hi=pct(S["citl"][m])[1],
                 brier_r1=brier_score_loss(y, r1[m]), brier_r2=brier_score_loss(y, r2[m]),
                 brier_r2_lo=pct(S["brier_r2"][m])[0], brier_r2_hi=pct(S["brier_r2"][m])[1],
                 slope_r2=sl2, citl_r2=ci2, threshold=thr[m], age_cut=age_cut)
        for ref in ["LR", BASE]:
            d = np.array(S["auc"][m]) - np.array(S["auc"][ref]); lo, hi = pct(d)
            r[f"dauc_vs_{ref}"] = auc - ref_auc[ref]; r[f"dauc_vs_{ref}_lo"] = lo; r[f"dauc_vs_{ref}_hi"] = hi
        rows.append(r)
        # decision analysis at pre-specified thresholds
        for kd, P in kinds.items():
            for t in tpre:
                v = NB[(m, kd, t)]; ref = np.array(NB[("LR", kd, t)]); allb = np.array(NA[t])
                nbp = net_benefit(y, P[m], t)
                nbrows.append(dict(scenario=scn, test=site, model=m, kind=kd, t=t, nb=nbp, nb_lo=pct(v)[0], nb_hi=pct(v)[1],
                                   nb_all=nb_all(y, t), d_vs_all=nbp - nb_all(y, t),
                                   d_vs_all_lo=pct(np.array(v) - allb)[0], d_vs_all_hi=pct(np.array(v) - allb)[1],
                                   d_vs_LR=nbp - net_benefit(y, P["LR"], t),
                                   d_vs_LR_lo=pct(np.array(v) - ref)[0], d_vs_LR_hi=pct(np.array(v) - ref)[1]))
        # decision curves + calibration bins (point estimates)
        for kd, P in kinds.items():
            for t in tgrid:
                curves.append(dict(scenario=scn, test=site, model=m, kind=kd, t=t, nb=net_benefit(y, P[m], t)))
        try:
            q = pd.qcut(preds[m], 10 if len(y) > 1000 else 5, duplicates="drop")
            g = pd.DataFrame({"p": preds[m], "y": y, "q": q}).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            for i, rr in enumerate(g.itertuples()):
                bins.append(dict(scenario=scn, test=site, model=m, bin=i + 1, mean_pred=rr.p, obs_rate=rr.y, n=rr.n))
        except Exception:
            pass
        # subgroup performance
        pr = preds[m] >= thr[m]
        pt = sub_diffs(y, pr, sex, age, age_cut)
        groups = {"male": sex == 1, "female": sex == 0, "younger": age < age_cut, "older": age >= age_cut}
        for gname, msk in groups.items():
            se, sp = sens_spec(y[msk], pr[msk]); ev = int(y[msk].sum()); ne = int((1 - y[msk]).sum())
            auc_g = roc_auc_score(y[msk], preds[m][msk]) if ev >= MIN_EVENTS and ne >= MIN_EVENTS else np.nan
            citl_g = calib_fast(y[msk], preds[m][msk])[1] if ev >= MIN_EVENTS and ne >= MIN_EVENTS else np.nan
            subrows.append(dict(scenario=scn, test=site, model=m, kind="group", group=gname, n=int(msk.sum()), events=ev,
                                prevalence=y[msk].mean() if msk.sum() else np.nan, sens=se, spec=sp, auc=auc_g, citl=citl_g))
        for k in D:
            subrows.append(dict(scenario=scn, test=site, model=m, kind="diff", group=k, est=pt[k],
                                lo=pct(D[k][m])[0] if np.isfinite(pt[k]) else np.nan, hi=pct(D[k][m])[1] if np.isfinite(pt[k]) else np.nan))
    return rows, nbrows, subrows, curves, bins


def auc_only(test, preds, B, rng, ref="LR"):
    y = test["y"].values; names = list(preds); A = {m: [] for m in names}
    for _ in range(B):
        ix = strat_idx(y, rng)
        for m in names: A[m].append(roc_auc_score(y[ix], preds[m][ix]))
    rows = []
    for m in names:
        lo, hi = pct(A[m]); d = np.array(A[m]) - np.array(A[ref]); dl, dh = pct(d)
        rows.append(dict(model=m, n=len(y), events=int(y.sum()), auc=roc_auc_score(y, preds[m]), auc_lo=lo, auc_hi=hi,
                         d_vs_ref=roc_auc_score(y, preds[m]) - roc_auc_score(y, preds[ref]), d_lo=dl, d_hi=dh, ref=ref))
    return rows


# ------------------------------------------------------------------------------ SHAP (descriptive only)
def shap_importance(model, train, test, feats, rng, n_bg=50, n_eval=100):
    import shap
    bg = train[feats].sample(min(n_bg, len(train)), random_state=int(rng.integers(1e9)))
    ev = test[feats].sample(min(n_eval, len(test)), random_state=int(rng.integers(1e9)))
    f = lambda X: model.predict_proba(pd.DataFrame(X, columns=feats))[:, 1]
    ex = shap.Explainer(f, shap.maskers.Independent(bg.values, max_samples=n_bg), algorithm="permutation", feature_names=feats)
    return np.abs(ex(ev.values, max_evals=2 * len(feats) + 1 + 20, silent=True).values).mean(0)


# ------------------------------------------------------------------------------ scenarios
def build_scenarios(a):
    uci = load_uci(a.uci_dir, a.include_swiss)
    path = a.framingham
    f1, f3 = load_fram(path, 1), load_fram(path, 3)
    cc = (lambda d: d.dropna(subset=FEATS).reset_index(drop=True)) if a.complete_case else (lambda d: d)
    uci = {k: cc(v) for k, v in uci.items()}
    ids = np.array(sorted(f1.id.unique())); np.random.default_rng(a.seed).shuffle(ids)
    dev_ids, test_ids = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])
    sp = lambda d, s: cc(d[d.id.isin(s)].reset_index(drop=True))
    dev, c0, c = sp(f1, dev_ids), sp(f1, test_ids), sp(f3, test_ids)
    fam = lambda site: "uci" if site in UCI_FILES else "fram"
    S = []
    for held in uci:
        S.append(dict(name=f"A_{held}", train=pd.concat([v for k, v in uci.items() if k != held], ignore_index=True),
                      tests={held: uci[held]}, ext=False, shap=True, full=True, family="uci"))
    pooled = pd.concat(uci.values(), ignore_index=True)
    S.append(dict(name="B1_UCI_to_Fram", train=pooled, tests={"Framingham exam 1": cc(f1)}, ext=False, shap=False, full=False, family="fram"))
    S.append(dict(name="B2_Fram_to_UCI", train=cc(f1), tests=dict(uci), ext=False, shap=False, full=False, family="uci"))
    S.append(dict(name="C0_Fram_disjoint", train=dev, tests={"Framingham exam 1 (other participants)": c0}, ext=True, shap=False, full=True, family="fram"))
    S.append(dict(name="C_Fram_temporal", train=dev, tests={"Framingham exam 3 (other participants)": c}, ext=True, shap=True, full=True, family="fram", frs=True))
    if not a.no_variants:
        for nm, oc in [("C_S1_composite", "composite"), ("C_S2_exclDeath", "excl_noncvd_death")]:
            d1, d3 = load_fram(path, 1, oc), load_fram(path, 3, oc)
            S.append(dict(name=nm, train=sp(d1, dev_ids), tests={f"Framingham exam 3 ({oc})": sp(d3, test_ids)}, ext=False, shap=False, full=False, family="fram"))
    return S, dict(uci=uci, f1=f1, f3=f3, dev=dev, c0=c0, c=c)


def describe(src):
    rows = []
    for k, d in src.items():
        rows.append({"Source": k, "n": len(d), "Age, mean (SD)": f"{d.age.mean():.1f} ({d.age.std():.1f})", "Male, %": round(100 * d.sex.mean(), 1),
                     "Cholesterol, mg/dL": f"{d.chol.mean():.1f} ({d.chol.std():.1f})", "Systolic BP, mmHg": f"{d.sbp.mean():.1f} ({d.sbp.std():.1f})",
                     "Hyperglycaemia, %": round(100 * d.hyperglycaemia.mean(), 1), "Events, n (%)": f"{int(d.y.sum())} ({100 * d.y.mean():.1f})",
                     "Missing predictor values, %": round(100 * d[FEATS].isna().mean().mean(), 1),
                     "n_female": int((d.sex == 0).sum()), "events_female": int(((d.sex == 0) & (d.y == 1)).sum())})
    return pd.DataFrame(rows)


def attrition(f1, f3):
    f1 = f1.copy(); f1["attended_exam3"] = f1.id.isin(set(f3.id)).astype(int)
    g = f1.groupby("attended_exam3").agg(n=("id", "size"), age=("age", "mean"), male=("sex", "mean"), chol=("chol", "mean"),
                                         sbp=("sbp", "mean"), hyperglycaemia=("hyperglycaemia", "mean"), smoker=("cursmoke", "mean"),
                                         chd10_exam1=("y", "mean"), death10_exam1=("death10", "mean")).reset_index()
    return g


# ------------------------------------------------------------------------------ stages
def run_main(a, S, aux):
    os.makedirs(f"{a.out}/partial", exist_ok=True)
    allrows = {k: [] for k in ["disc", "nb", "sub", "curves", "bins", "shap", "frs", "bp", "preds"]}
    B = 20 if FAST else a.B
    for scn in S:
        pk = f"{a.out}/partial/{scn['name']}.pkl"
        if a.resume and os.path.exists(pk):
            R = pickle.load(open(pk, "rb"))
        else:
            t0 = time.time(); rng = np.random.default_rng(a.seed + zlib.crc32(scn["name"].encode()) % 1000)
            tr = scn["train"]; print(f"[{scn['name']}] train n={len(tr)} events={int(tr.y.sum())}", flush=True)
            fitted, oof, bp = fit_models(tr, a.seed, scn["ext"])
            names = list(fitted); thr = {m: youden(tr.y.values, oof[m]) for m in names}
            internal = {m: roc_auc_score(tr.y, oof[m]) for m in names}; age_cut = float(tr.age.median())
            R = {k: [] for k in allrows}; R["bp"] = {"scenario": scn["name"], "params": bp, "ext": scn["ext"]}
            for site, te in scn["tests"].items():
                preds = {m: fitted[m].predict_proba(te[FMAP[m]])[:, 1] for m in names}
                rows, nbr, sbr, cur, bins = evaluate(scn["name"], site, te, preds, thr, age_cut, scn["family"], B, rng, a.seed)
                for r in rows:
                    r.update(internal_cv_auc=internal[r["model"]], auc_drop=internal[r["model"]] - r["auc"], n_train=len(tr), events_train=int(tr.y.sum()))
                R["disc"] += rows; R["nb"] += nbr; R["sub"] += sbr; R["curves"] += cur; R["bins"] += bins
                R["preds"].append(pd.DataFrame({"scenario": scn["name"], "test": site, "y": te.y.values, "age": te.age.values, "sex": te.sex.values,
                                                **{f"p_{m}": preds[m] for m in names}}))
                if scn["shap"] and not a.no_shap:
                    for m in MODELS:
                        v = shap_importance(fitted[m], tr, te, FMAP[m], rng)
                        R["shap"].append({"scenario": scn["name"], "test": site, "model": m, **dict(zip(FEATS, v))})
                if scn.get("frs"):
                    need = ["age", "sex", "chol", "sbp", "hdl", "bpmeds", "cursmoke", "diabetes"]
                    sub = te[te[need].notna().all(axis=1)].reset_index(drop=True)
                    ps = {m: fitted[m].predict_proba(sub[FMAP[m]])[:, 1] for m in names}; ps["FRS"] = frs_cvd(sub)
                    for r in auc_only(sub, ps, B, rng, ref="FRS"): r.update(scenario=scn["name"], test=site, subset="complete FRS inputs"); R["frs"].append(r)
            pickle.dump(R, open(pk, "wb")); print(f"   done in {time.time() - t0:.0f}s", flush=True)
        for k in allrows:
            if k == "bp": allrows["bp"].append(R["bp"])
            else: allrows[k] += R[k] if k != "preds" else R[k]
    pd.DataFrame(allrows["disc"]).to_csv(f"{a.out}/table_discrimination_calibration.csv", index=False)
    pd.DataFrame(allrows["nb"]).to_csv(f"{a.out}/table_net_benefit.csv", index=False)
    pd.DataFrame(allrows["sub"]).to_csv(f"{a.out}/table_subgroups.csv", index=False)
    pd.DataFrame(allrows["curves"]).to_csv(f"{a.out}/dca_curves.csv", index=False)
    pd.DataFrame(allrows["bins"]).to_csv(f"{a.out}/calibration_bins.csv", index=False)
    pd.DataFrame(allrows["shap"]).to_csv(f"{a.out}/shap_mean_abs.csv", index=False)
    pd.DataFrame(allrows["frs"]).to_csv(f"{a.out}/table_frs_benchmark.csv", index=False)
    pd.concat(allrows["preds"]).to_csv(f"{a.out}/predictions.csv.gz", index=False)
    json.dump(allrows["bp"], open(f"{a.out}/best_params.json", "w"), indent=1, default=str)


def _one_boot(seed_b, train, test, bp, names):
    rng = np.random.default_rng(seed_b)
    tr = train.iloc[strat_idx(train.y.values, rng)]; te = test.iloc[strat_idx(test.y.values, rng)]
    grid = model_grid(0); out = {}
    for m in names:
        try:
            p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in bp[m].items()}
            est = clone(grid[m][0]).set_params(**p); est.fit(tr[FMAP[m]], tr.y.values)
            out[m] = roc_auc_score(te.y.values, est.predict_proba(te[FMAP[m]])[:, 1])
        except Exception:
            out[m] = np.nan
    return out


def run_fullboot(a, S):
    bps = {b["scenario"]: b for b in json.load(open(f"{a.out}/best_params.json"))}
    B = 3 if FAST else a.Bfull; rows = []; os.makedirs(f"{a.out}/fullboot", exist_ok=True)
    store = {}
    for scn in S:
        if not scn["full"]: continue
        names = list(bps[scn["name"]]["params"])
        for site, te in scn["tests"].items():
            f = f"{a.out}/fullboot/{scn['name']}__{site.replace(' ', '_')}.pkl"
            if a.resume and os.path.exists(f):
                res = pickle.load(open(f, "rb"))
            else:
                t0 = time.time(); print(f"[fullboot {scn['name']} / {site}] B={B}", flush=True)
                res = Parallel(n_jobs=2)(delayed(_one_boot)(a.seed * 1000 + b, scn["train"], te, bps[scn["name"]]["params"], names) for b in range(B))
                pickle.dump(res, open(f, "wb")); print(f"   done in {time.time() - t0:.0f}s", flush=True)
            store[(scn["name"], site)] = pd.DataFrame(res)
    summ = []
    def summarise(label, site, df):
        for m in df.columns:
            if m in ("LR",): continue
            for ref in ["LR", BASE]:
                if m == ref: continue
                d = (df[m] - df[ref]).values; lo, hi = pct(d)
                r = dict(scenario=label, test=site, model=m, ref=ref, auc_mean=df[m].mean(), auc_lo=pct(df[m])[0], auc_hi=pct(df[m])[1],
                         d_mean=np.nanmean(d), d_lo=lo, d_hi=hi, B=int(np.isfinite(d).sum()))
                for mg in (0.02, 0.03, 0.05):
                    r[f"LR_noninf_{int(mg * 100)}"] = bool(hi < mg) if ref == "LR" else np.nan
                    r[f"ML_noninf_{int(mg * 100)}"] = bool(lo > -mg) if ref == "LR" else np.nan
                summ.append(r)
    for (scn, site), df in store.items(): summarise(scn, site, df)
    pd.DataFrame(summ).to_csv(f"{a.out}/table_fullboot_noninferiority.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci_dir", required=True); ap.add_argument("--framingham", required=True)
    ap.add_argument("--out", default="results"); ap.add_argument("--stage", default="all", choices=["main", "fullboot", "all"])
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--Bfull", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--complete_case", action="store_true"); ap.add_argument("--include_swiss", action="store_true")
    ap.add_argument("--no_shap", action="store_true"); ap.add_argument("--no_variants", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    global FAST; FAST = a.fast
    os.makedirs(a.out, exist_ok=True)
    S, aux = build_scenarios(a)
    describe({**aux["uci"], "Framingham exam 1": aux["f1"], "Framingham development half (exam 1)": aux["dev"],
              "Framingham test half, exam 1": aux["c0"], "Framingham test half, exam 3": aux["c"]}).to_csv(f"{a.out}/table1_characteristics.csv", index=False)
    attrition(aux["f1"], aux["f3"]).to_csv(f"{a.out}/attrition_exam3.csv", index=False)
    if a.stage in ("main", "all"): run_main(a, S, aux)
    if a.stage in ("fullboot", "all"): run_fullboot(a, S)
    json.dump({"python": sys.version, "platform": platform.platform(), "seed": a.seed, "B": a.B, "Bfull": a.Bfull,
               "complete_case": a.complete_case, "include_swiss": a.include_swiss, "numpy": np.__version__, "pandas": pd.__version__,
               "sklearn": __import__("sklearn").__version__, "xgboost": __import__("xgboost").__version__, "shap": __import__("shap").__version__},
              open(f"{a.out}/environment.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
