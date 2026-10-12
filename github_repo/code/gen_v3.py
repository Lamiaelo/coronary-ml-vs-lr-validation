"""Generate blocks_v3.json (article content) directly from the result CSVs, so that every number is traceable."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_v3 import load, SET, ML, SIX

R = sys.argv[1]; OUT = sys.argv[2]
RCC = sys.argv[3] if len(sys.argv) > 3 else None
RSW = sys.argv[4] if len(sys.argv) > 4 else None
G = load(R); d, nb, sg, fr, fb, sh, t1, att = (G[k] for k in ["d", "nb", "sg", "fr", "fb", "sh", "t1", "att"])

MN = "−"
def f(x, k=3):
    if pd.isna(x): return "–"
    r = f"{x:.{k}f}"
    if float(r) == 0: r = r.lstrip("-")
    return r.replace("-", MN)
def fci(est, lo, hi, k=3): return f"{f(est,k)} ({f(lo,k)} to {f(hi,k)})"
def row(s, m): return d[(d.scenario == s) & (d.model == m)].iloc[0]
def auc_ci(s, m):
    r = row(s, m); return f"{f(r.auc)} ({f(r.auc_lo)}–{f(r.auc_hi)})"
LAB = {"A_Cleveland": "Cleveland", "A_Hungary": "Hungary", "A_VA": "VA", "B1_UCI_to_Fram": "UCI to Framingham",
       "C0_Fram_disjoint": "Framingham, same exam", "C_Fram_temporal": "Framingham, exam 3"}
MNAME = {"FRS": "Framingham 2008 equation", "LR": "LR", "RF": "RF", "GB": "GB", "SVM": "SVM", "MLP": "MLP", "XGB": "XGBoost", "AGE": "Age only", "LRX": "Extended LR"}

# ---------------------------------------------------------------- derived statistics
ml = d[d.scenario.isin(SET) & d.model.isin(ML)]
n_cmp = len(ml); n_better = int((ml.dauc_vs_LR_lo > 0).sum()); n_worse = int((ml.dauc_vs_LR_hi < 0).sum())
n_ptneg = int((ml.dauc_vs_LR < 0).sum())
assert n_better == 0, "text assumes no ML model significantly better than LR"
six = d[d.scenario.isin(SET) & d.model.isin(SIX)]
auc_min, auc_max = six.auc.min(), six.auc.max()
rng_set = {s: (six[six.scenario == s].auc.min(), six[six.scenario == s].auc.max()) for s in SET}
age_row = d[d.scenario.isin(SET) & (d.model == "AGE")]
best_ext = {s: six[six.scenario == s].sort_values("auc").iloc[-1].model for s in SET}
best_int = {s: six[six.scenario == s].sort_values("internal_cv_auc").iloc[-1].model for s in SET}
n_same = sum(best_ext[s] == best_int[s] for s in SET)
lr_int_best = sum(best_int[s] == "LR" for s in SET)
drop_lr = {s: row(s, "LR").auc_drop for s in SET}
a_ex = [s for s in SET if s.startswith("A_") or s.startswith("B1") or s.startswith("C_")]
vs_age_ok = int((six.dauc_vs_AGE_lo > 0).sum()) ; vs_age_n = int(((six.model != "AGE")).sum())
fbm = fb[(fb.ref == "LR") & fb.model.isin(ML)]
def nic(sc, col): return int(fbm[fbm.scenario == sc][col].sum())
pooled = fbm[fbm.scenario == "A_pooled_mean_of_sites"]
sgd = sg[(sg.kind == "diff") & sg.scenario.isin(SET) & sg.model.isin(SIX)]
def sgsum(k):
    z = sgd[(sgd.group == k) & sgd.est.notna()]
    return len(z), int(((z.lo > 0) | (z.hi < 0)).sum()), z.est.min(), z.est.max()
sx = sgsum("dsens_sex"); sxs = sgsum("dspec_sex"); ag = sgsum("dsens_age"); ags = sgsum("dspec_age")

frs = fr
frs_ref = frs[frs.model == "FRS"].iloc[0]
def frr(m): return frs[frs.model == m].iloc[0]
lr_frs = frr("LR"); lrx_frs = frr("LRX")

_tdev = t1[t1.Source.str.contains("development half")].iloc[0]; _tte = t1[t1.Source.str.contains("test half, exam 3")].iloc[0]
age_tr = _tdev["Age, mean (SD)"].split(" ")[0]; age_te = _tte["Age, mean (SD)"].split(" ")[0]
hg_tr = float(_tdev["Hyperglycaemia, %"]); hg_te = float(_tte["Hyperglycaemia, %"])
# ---------------------------------------------------------------- block helpers
K = []
P = lambda x: K.append({"t": "p", "x": x})
PL = lambda l, x: K.append({"t": "pl", "l": l, "x": x})
H1 = lambda x: K.append({"t": "h1", "x": x})
H2 = lambda x: K.append({"t": "h2", "x": x})
LI = lambda x: K.append({"t": "li", "x": x})
def TB(cap, widths, rows, note=None):
    K.append({"t": "cap", "x": cap}); K.append({"t": "tbl", "w": widths, "rows": rows})
    if note: K.append({"t": "note", "x": note})
def IMG(file, name, note):
    K.append({"t": "img", "file": file, "name": name}); K.append({"t": "note", "x": note})

# ================================================================ 1. Introduction
H1("1. Introduction")
P("Cardiovascular disease remains the leading cause of death worldwide [@roth]. Risk prediction models are used to decide who should receive preventive treatment or further testing, and a large literature has accumulated on conventional regression-based risk scores [@damen]. Over the last decade many studies have reported that machine learning (ML) algorithms can predict cardiovascular events more accurately than established scores [@weng], and the public availability of small benchmark datasets has made it easy to publish such comparisons. Systematic reviews, however, find no consistent performance benefit of ML over logistic regression (LR) in clinical prediction once the studies are judged on the quality of their validation [@christ,@nusin], and systematic reviews of prediction models in other conditions found most to be at high risk of bias [@wynants]. Data leakage between development and evaluation data is a recurrent source of over-optimistic results [@kapoor].")
P("What matters for use is not performance in the development sample but performance in the setting where the model will be applied. Validation in other sites, populations and time periods is therefore recommended [@steyer,@riley], and its interpretation requires separating differences in case mix from differences in the true predictor effects [@debray]. In practice, a model that discriminates well can still give predicted risks that are systematically too high or too low, and this miscalibration is often the main reason a model fails after transfer [@vancal,@steyerberg10]. Discrimination and calibration also do not tell whether using a model improves decisions; decision-curve analysis addresses this by weighing true and false positives at clinically relevant risk thresholds [@vickers06,@vickers16]. Changes in the underlying data distribution between development and deployment are a known threat to clinical models [@subb,@finlayson].")
P("Several gaps follow. First, ML algorithms and LR are seldom compared in the same study at held-out sites and over time with paired uncertainty intervals and with an age-only benchmark that shows how much the other predictors add. Second, calibration after transfer, the effect of simple recalibration, and net benefit are rarely reported together with discrimination for the same models. Third, claims that LR is “as good as” ML are usually informal, whereas a non-inferiority claim requires an explicit margin and a statement of which sources of variability are included [@piaggio]. These gaps matter for analysts and decision makers who must choose a model for a new setting with a limited set of routinely recorded predictors.")
P("The objective of this study is to compare LR with five ML algorithms (random forest, gradient boosting, support vector machine, multilayer perceptron and XGBoost) for coronary risk prediction from five common predictors under leakage-controlled multi-site and temporal validation. We ask: (Q1) do ML models outperform LR at held-out sites and at a later examination; (Q2) is LR non-inferior to ML under explicit margins when training variability is included; (Q3) how well calibrated are the predictions after transfer, and how much does simple recalibration help; (Q4) do the models provide net benefit over default strategies and over an age-only benchmark; and (Q5) does sensitivity at a fixed threshold differ by sex and age. The study is an exploratory methodological evaluation on small, old public datasets and does not propose a deployable risk score.")

# ================================================================ 2. Methods
H1("2. Materials and Methods")
H2("2.1 Data Sources")
P("We used two public sources. The UCI Heart Disease repository contains patients referred for coronary angiography at four centres [@detrano,@janosi]; we analysed Cleveland (n = 303), Hungary (n = 294) and the Veterans Affairs Long Beach centre (VA, n = 200). The Swiss site (n = 123) was excluded from the main analysis because cholesterol is recorded as zero (missing) for almost all patients; it is included in a sensitivity analysis. The outcome is angiographic coronary disease (the repository variable num > 0), which is a diagnostic outcome measured at a single time point. The second source is the Framingham Heart Study teaching dataset [@dawber,@nhlbi], which contains repeated examinations of community participants with follow-up for incident events. The outcome there is incident coronary heart disease (CHD) within 10 years (3,652.5 days) after an examination, defined as any CHD event (variable ANYCHD) occurring after the examination date, among participants free of CHD at that examination.")
P("Both sources are de-identified public data; no new data were collected and no ethical approval was required. The Framingham file analysed here (frmgham2.csv) was obtained from a public GitHub copy of the NHLBI teaching dataset, which contains a random subset of participants and is not suitable for generating population estimates. The original should be requested from BioLINCC to reproduce the work.")
H2("2.2 Predictors and Harmonisation")
P("Only variables available in both sources were used: age (years), sex, total cholesterol (mg/dL), systolic blood pressure (mmHg) and hyperglycaemia. Hyperglycaemia was defined in UCI as fasting blood sugar above 120 mg/dL and in Framingham as glucose above 120 mg/dL or diagnosed diabetes. Because Framingham glucose was not always measured in the fasting state, this variable is not identical across sources, and we treat it as the least reliable predictor. Cholesterol or blood pressure recorded as zero in UCI was set to missing. Missing values were imputed with the training-set median inside each model pipeline, so that no information from the test data was used.")
H2("2.3 Validation Settings")
P("Figure 1 and Table 2 summarise the settings. In setting A (leave-one-site-out), models were trained on two UCI sites and tested on the third, giving three tests with the same outcome definition. In setting C (temporal), participants in Framingham examination 1 were split at random by participant identifier into a development half and a test half. Models were trained on the development half at examination 1 and tested on the other half of participants at examination 3, about 12 years later. Participants in the two halves never overlap, so no person contributes to both training and test data. Setting C0 uses the same development half and tests on the other half at examination 1; it separates the effect of new participants from the effect of time. Setting B1 trains on all UCI sites pooled and tests on Framingham examination 1, and B2 trains on Framingham examination 1 and tests on each UCI site. In B1 and B2 the outcome definition differs between development and test data (angiographic disease versus 10-year incident CHD) and the event prevalence differs several-fold, so they are stress tests of transfer and not estimates of clinical performance. Results of B2 are reported in the Appendix.")
H2("2.4 Models and Benchmarks")
P("The compared algorithms were LR with L2 penalty, random forest (RF), gradient boosting (GB), an RBF support vector machine (SVM) with balanced class weights and sigmoid (Platt) calibration, a multilayer perceptron (MLP) with L2 regularisation and no early stopping, and XGBoost [@xgb]. All were implemented with scikit-learn [@sklearn] and XGBoost. Hyper-parameters were chosen by five-fold cross-validated AUC within the training data (grids are in the supplementary code). A decision threshold for sensitivity and specificity was chosen once, using the Youden index on out-of-fold training predictions, and then applied unchanged to test data. Three reference models were added. The age-only LR benchmark shows how much of the discrimination comes from age alone. An extended LR (LRX) adds smoking status, cigarettes per day, antihypertensive treatment, diabetes and body mass index; it was fitted only in the Framingham settings, where these variables exist. Finally, for the temporal test we computed the published Framingham general cardiovascular risk equation [@dagostino] without refitting, on participants with complete inputs (including HDL cholesterol). This equation predicts general cardiovascular disease and not CHD only, and uses more predictors, so it is used only as a discrimination benchmark. Its coefficients were checked against two independent public implementations.")
H2("2.5 Performance Measures")
P("Discrimination was measured by the area under the receiver operating characteristic curve (AUC). Calibration was summarised by the calibration slope and the calibration-in-the-large (CITL, the intercept when the slope is fixed at 1) [@vancal,@steyerberg10] and by the Brier score. To show what a simple local update achieves, predictions were recalibrated within the target data by two methods, each cross-fitted with five folds so that no observation is used to recalibrate its own prediction: an intercept update (R1) and an intercept and slope update (R2). These estimates describe what is achievable if labelled outcome data from the new site are available (the cross-fitting uses 80% of the test data, from 160 patients at the VA site to several thousand in Framingham), and they are not an out-of-the-box performance.")
P("Clinical usefulness was assessed by decision-curve analysis [@vickers06,@vickers16]. Net benefit was computed for recalibrated (R2) predictions at pre-specified thresholds: 0.30 and 0.50 for the UCI sites, where the prevalence of angiographic disease is high, and 0.10 and 0.20 for 10-year CHD in Framingham. The comparators were treating everyone, treating no one and the age-only model. Subgroup performance was summarised by the difference in sensitivity and in specificity at the fixed threshold between men and women and between participants younger and older than the training median age (signed differences; sensitivity differences are the equal opportunity difference of [@hardt]). Sensitivity was evaluated only for groups with at least 20 events and specificity only for groups with at least 20 non-events. These differences are descriptive: unequal base rates in the groups lead to unequal selection rates for any model that uses age and sex.")
H2("2.6 Uncertainty and Non-Inferiority")
P("Uncertainty intervals for test-set measures are percentile intervals from 1,000 ordinary bootstrap resamples of the test patients with the fitted models held fixed (resampling is not stratified by outcome, so that the variability of the event rate, which drives calibration-in-the-large and net benefit, is included). Differences between models (ΔAUC) are paired on the same resamples. These intervals reflect sampling variation of the test data only. To include variation due to training, we also ran a full bootstrap (300 resamples, stratified by outcome) that resamples both the training and the test data and refits every model with the hyper-parameters fixed at their selected values. This is computationally cheaper than repeating the tuning and may understate training variability slightly; duplicated training rows can also affect the inner folds of the SVM calibration. Intervals for recalibrated measures resample the cross-fitted predictions and do not repeat the recalibration. From this bootstrap we assessed non-inferiority of LR relative to each ML model, defined as an upper confidence limit of the AUC advantage of the ML model below a margin, for margins of 0.02, 0.03 and 0.05. The margins were chosen after seeing the data, were not pre-registered and have no clinical derivation; we therefore report the result for all three and make no claim of equivalence. For symmetry we also report whether each ML model is non-inferior to LR (lower limit above minus the margin). Analyses were not adjusted for multiplicity and should be read as exploratory.")
H2("2.7 Sensitivity Analyses")
P("We repeated setting C with two other outcome definitions: a composite of CHD or death from any cause within 10 years, and the CHD outcome after excluding participants who died of non-CHD causes within 10 years (which avoids treating these deaths as non-events but selects survivors; see [@austin] for competing risks). The exclusion is applied to both development and test data. We compared Framingham participants with and without a visit at examination 3 to describe survivor and attendance selection. We re-ran the primary analysis on complete cases only and with the Swiss site added. Descriptive permutation-based Shapley additive explanations (SHAP) [@lundberg17] are in the Appendix; they were computed on the test data and are not used for any conclusion about model stability.")
H2("2.8 Reproducibility and Reporting")
P("The analysis script takes the public files as input and writes all result tables (Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.52.0; random seed 2026). The figures and formatted tables are produced by helper scripts that read the result files. The study follows TRIPOD+AI [@tripodai,@tripod]; a reporting map and a PROBAST [@probast] self-assessment are in the Appendix.")

# ================================================================ 3. Results
H1("3. Results")
H2("3.1 Cohorts and Settings")
t1rows = [["Source", "n", "Age, mean (SD)", "Male, %", "Cholesterol, mg/dL", "SBP, mmHg", "Hyperglyc., %", "Events, n (%)", "Missing, %"]]
for _, r in t1.iterrows():
    t1rows.append([r["Source"], f'{int(r["n"]):,}', r["Age, mean (SD)"], f'{r["Male, %"]:.1f}', r["Cholesterol, mg/dL"], r["Systolic BP, mmHg"], f'{r["Hyperglycaemia, %"]:.1f}', r["Events, n (%)"], f'{r["Missing predictor values, %"]:.1f}'])
n_f1 = int(t1[t1.Source == "Framingham exam 1"].n.iloc[0]); ev_f1 = t1[t1.Source == "Framingham exam 1"]["Events, n (%)"].iloc[0]
a_att = att.set_index("attended_exam3")
P(f"Table 1 describes the cohorts. The three UCI sites differ markedly: the VA site is almost entirely male (97%), older and has a 74.5% prevalence of angiographic disease, against 45.9% in Cleveland and 36.1% in Hungary. Framingham examination 1 included {n_f1:,} participants free of CHD, of whom {ev_f1.split(' ')[0]} ({ev_f1.split('(')[1].rstrip(')')}%) had a CHD event within 10 years. At examination 3 the test half had 1,437 participants and 14.1% events. The examination 3 test sample is, by construction, restricted to participants who were alive, attended and were free of CHD at examination 3. Among all examination 1 participants, those included at examination 3 (n = {int(a_att.loc[1,'n']):,}) were younger at examination 1 than those not included (n = {int(a_att.loc[0,'n']):,}; mean age {a_att.loc[1,'age']:.1f} vs {a_att.loc[0,'age']:.1f} years), while {100*a_att.loc[0,'chd10_exam1']:.0f}% of the others had had a CHD event and {100*a_att.loc[0,'death10_exam1']:.0f}% had died within 10 years of examination 1; the sample is therefore a healthier, survivor-selected subset of the examination 1 sample (Table A5).")
TB("Table 1. Characteristics of the datasets", [1500, 700, 1000, 650, 1100, 900, 1150, 1000, 800], t1rows,
   "Note: SBP, systolic blood pressure. Missing is the mean percentage of missing values among the five predictors. The Framingham development half and test half are the participant-disjoint subsets used in settings C0 and C.")
setrows = [["Setting", "Development data", "Test data", "Outcome (development / test)", "Train n (events)", "Test n (events)"]]
bpn = d.drop_duplicates("scenario").set_index("scenario")
def ne(s, m="LR"):
    r = row(s, m); return f"{int(r.n):,} ({int(r.events)})"
def tr_ne(s):
    r = row(s, "LR"); return f"{int(r.n_train):,} ({int(r.events_train)})"
setrows += [["A: Cleveland", "Hungary + VA", "Cleveland", "Angiographic / same", tr_ne("A_Cleveland"), ne("A_Cleveland")],
            ["A: Hungary", "Cleveland + VA", "Hungary", "Angiographic / same", tr_ne("A_Hungary"), ne("A_Hungary")],
            ["A: VA", "Cleveland + Hungary", "VA", "Angiographic / same", tr_ne("A_VA"), ne("A_VA")],
            ["B1", "UCI pooled", "Framingham exam 1", "Angiographic / 10-year CHD", tr_ne("B1_UCI_to_Fram"), ne("B1_UCI_to_Fram")],
            ["C0", "Framingham exam 1, half 1", "Framingham exam 1, half 2", "10-year CHD / same", tr_ne("C0_Fram_disjoint"), ne("C0_Fram_disjoint")],
            ["C", "Framingham exam 1, half 1", "Framingham exam 3, half 2", "10-year CHD / same", tr_ne("C_Fram_temporal"), ne("C_Fram_temporal")]]
TB("Table 2. Validation settings", [1250, 1900, 1900, 2000, 1300, 1350], setrows,
   "Note: B2 (Framingham to each UCI site) is reported in the Appendix. Events are cases of the outcome in the respective set.")
IMG("fig1_design_v3.png", "Study design", "Figure 1. Validation settings. Arrows go from development data to test data; the outcome definition changes only in B1 and B2")

H2("3.2 Discrimination")
lr_age = d[(d.scenario.isin(SET)) & (d.model == "LR")]
ex_ = lr_age[~(lr_age.dauc_vs_AGE_lo > 0)]
vs_age_txt = ("LR exceeded it with a 95% interval excluding zero in all six settings" if ex_.empty else
              "LR exceeded it with a 95% interval excluding zero in all settings except " + ", ".join(f"{LAB[r.scenario]} (\u0394AUC {f(r.dauc_vs_AGE)}, 95% CI {f(r.dauc_vs_AGE_lo)} to {f(r.dauc_vs_AGE_hi)})" for r in ex_.itertuples()))
sig_w = ml[ml.dauc_vs_LR_hi < 0]
sig_txt = "; ".join(f"{MNAME[r.model]} in {LAB[r.scenario]}" for r in sig_w.itertuples())
P(f"Across the six models and six settings, test-set AUCs ranged from {f(auc_min)} to {f(auc_max)} (Tables 3 and 4, Figure 2). In the UCI held-out sites the AUC ranges were {f(rng_set['A_Cleveland'][0])}–{f(rng_set['A_Cleveland'][1])} (Cleveland), {f(rng_set['A_Hungary'][0])}–{f(rng_set['A_Hungary'][1])} (Hungary) and {f(rng_set['A_VA'][0])}–{f(rng_set['A_VA'][1])} (VA). In Framingham they were {f(rng_set['C0_Fram_disjoint'][0])}–{f(rng_set['C0_Fram_disjoint'][1])} for new participants at the same examination (C0) and {f(rng_set['C_Fram_temporal'][0])}–{f(rng_set['C_Fram_temporal'][1])} at examination 3 (C). The age-only model reached {f(row('C_Fram_temporal','AGE').auc)} in C and {f(row('A_Cleveland','AGE').auc)}, {f(row('A_Hungary','AGE').auc)} and {f(row('A_VA','AGE').auc)} in the three UCI sites; {vs_age_txt}.")
def auc_table(sets, extra=None):
    hdr = ["Model"] + [LAB[s] for s in sets]
    rows = [hdr]
    for m in ["LR", "RF", "GB", "SVM", "MLP", "XGB", "AGE", "LRX"]:
        if m == "LRX" and not any(((d.scenario == s) & (d.model == m)).any() for s in sets): continue
        r = [MNAME[m]]
        for s in sets:
            r.append(auc_ci(s, m) if ((d.scenario == s) & (d.model == m)).any() else "–")
        rows.append(r)
    return rows
TB("Table 3. Test-set AUC (95% bootstrap interval) at held-out UCI sites", [1500, 2750, 2750, 2700], auc_table(["A_Cleveland", "A_Hungary", "A_VA"]),
   "Note: LR, logistic regression; RF, random forest; GB, gradient boosting; SVM, support vector machine; MLP, multilayer perceptron. Intervals reflect test-set sampling only.")
TB("Table 4. Test-set AUC (95% bootstrap interval) in the Framingham settings", [1500, 2750, 2750, 2700], auc_table(["B1_UCI_to_Fram", "C0_Fram_disjoint", "C_Fram_temporal"]),
   "Note: Extended LR was fitted only in the Framingham-trained settings (C0 and C). In B1 the outcome definition differs between training and test data.")
IMG("fig_auc.png", "AUC by model and setting", "Figure 2. Test-set AUC with 95% bootstrap intervals by model and setting")
def dtab():
    rows = [["Model vs LR"] + [LAB[s] for s in SET]]
    for m in ML:
        r = [MNAME[m]]
        for s in SET:
            x = row(s, m); mark = "†" if (x.dauc_vs_LR_lo > 0 or x.dauc_vs_LR_hi < 0) else ""
            r.append(f(x.dauc_vs_LR) + mark)
        rows.append(r)
    return rows
P(f"Paired comparisons with LR are in Table 5. Of {n_cmp} comparisons (five ML models in six settings, including the cross-outcome setting B1), none showed an ML model significantly better than LR, {n_worse} showed LR significantly better (95% interval of ΔAUC entirely below zero) and the point estimate favoured LR in {n_ptneg}. The significant differences were: {sig_txt}. The largest advantage of any ML model over LR was {f(ml.dauc_vs_LR.max())} AUC units ({MNAME[ml.loc[ml.dauc_vs_LR.idxmax(),'model']]}, {LAB[ml.loc[ml.dauc_vs_LR.idxmax(),'scenario']]}), with a 95% interval including zero. The extended LR with five additional predictors gave an AUC of {f(row('C0_Fram_disjoint','LRX').auc)} in C0 and {f(row('C_Fram_temporal','LRX').auc)} in C, a gain over LR of {f(row('C0_Fram_disjoint','LRX').dauc_vs_LR)} and {f(row('C_Fram_temporal','LRX').dauc_vs_LR)} (95% intervals {f(row('C0_Fram_disjoint','LRX').dauc_vs_LR_lo)} to {f(row('C0_Fram_disjoint','LRX').dauc_vs_LR_hi)} and {f(row('C_Fram_temporal','LRX').dauc_vs_LR_lo)} to {f(row('C_Fram_temporal','LRX').dauc_vs_LR_hi)}). The point estimates suggest that adding predictors helped at least as much as changing the algorithm, but the lower interval limits are at or below zero and the gain is small.")
TB("Table 5. Paired difference in AUC relative to logistic regression (ML minus LR)", [1600, 1350, 1250, 1100, 1450, 1500, 1450], dtab(),
   "Note: † the 95% bootstrap interval of the difference excludes zero. All such differences are negative (LR better). Full intervals are in Table A1.")

H2("3.3 Internal Versus External Performance")
P(f"The cross-validated AUC in the training data exceeded the external AUC in most settings. For LR the difference was {f(drop_lr['A_Cleveland'])} (Cleveland), {f(drop_lr['A_Hungary'])} (Hungary), {f(drop_lr['A_VA'])} (VA), {f(drop_lr['B1_UCI_to_Fram'])} (B1), {f(drop_lr['C0_Fram_disjoint'])} (C0) and {f(drop_lr['C_Fram_temporal'])} (C). The largest drop occurred in the temporal test (C), which may reflect changes in case mix over time, survivor selection, or both. The negative value in C0 means that the external AUC was higher than the cross-validated AUC; since the test half is a random split of the same examination, this is consistent with sampling variation, and the cross-validated AUC is itself slightly optimistic because the hyper-parameters were chosen on the same folds. LR had the highest internal AUC in {lr_int_best} of 6 settings, and the algorithm with the highest internal AUC also had the highest external AUC in {n_same} of 6 settings. The differences between models were small relative to their uncertainty, so these data do not support choosing a model by internal validation alone (Table A1 lists internal AUCs). In setting C the test participants were also on average about ten years older than the training participants ({age_te} vs {age_tr} years), which requires extrapolation in age (tree-based models cannot extrapolate), and the prevalence of hyperglycaemia rose from {hg_tr:.1f}% to {hg_te:.1f}% under a definition that is not strictly fasting; both factors, besides survivor selection, may contribute to the larger drop.")

H2("3.4 Non-Inferiority With Training Variability")
fb_rows = [["Setting", "LR non-inferior to ML, margin 0.02", "0.03", "0.05", "ML non-inferior to LR, margin 0.02", "0.03", "0.05"]]
for sc, lab in [("A_Cleveland", "Cleveland"), ("A_Hungary", "Hungary"), ("A_VA", "VA"), ("C0_Fram_disjoint", "C0"), ("C_Fram_temporal", "C")]:
    fb_rows.append([lab] + [f"{nic(sc, c)}/5" for c in ["LR_noninf_2", "LR_noninf_3", "LR_noninf_5", "ML_noninf_2", "ML_noninf_3", "ML_noninf_5"]])
assert all(nic(sc,"LR_noninf_5")==5 for sc in ["A_Cleveland","A_Hungary","C0_Fram_disjoint","C_Fram_temporal"]) and nic("A_VA","LR_noninf_5")==4
assert all(nic(sc,"LR_noninf_3")==5 for sc in ["A_Hungary","C0_Fram_disjoint","C_Fram_temporal"]) and nic("A_Cleveland","LR_noninf_3")<5 and nic("A_VA","LR_noninf_3")<5
assert nic("A_Hungary","LR_noninf_2")==5 and nic("C0_Fram_disjoint","LR_noninf_2")==5 and all(nic(sc,"LR_noninf_2")<5 for sc in ["A_Cleveland","A_VA","C_Fram_temporal"])
assert (fbm.d_lo>0).sum()==0
P(f"In the full bootstrap, which includes variability from training, the 95% intervals of ΔAUC were much wider than in the test-set bootstrap and none of the ML models was significantly better than LR in any setting (Table 6). LR was non-inferior to all five ML models at a margin of 0.05 in every setting except VA, where it was non-inferior to four. At a margin of 0.03 the same was true for Hungary, C0 and C, but not for Cleveland ({nic('A_Cleveland','LR_noninf_3')} of 5) or VA ({nic('A_VA','LR_noninf_3')} of 5). At the stricter margin of 0.02 non-inferiority was shown for all five models only in Hungary and C0. The reverse claim, that ML is non-inferior to LR, was rarely supported (for example {nic('A_Hungary','ML_noninf_5')} of 5 in Hungary and {nic('A_VA','ML_noninf_5')} of 5 in VA at a margin of 0.05). The data are therefore compatible with no gain from ML, but the full-bootstrap intervals of \u0394AUC were wide (lower limits down to {f(fbm.d_lo.min(),2)} and upper limits up to {f(fbm.d_hi.max(),2)}), so differences of several hundredths in either direction cannot be excluded, in particular in the smallest cohorts; equivalence is not established.")
TB("Table 6. Number of ML models (out of five) satisfying non-inferiority from the full bootstrap", [2250, 1550, 650, 650, 1550, 650, 650], fb_rows,
   "Note: Non-inferiority of LR requires the upper 95% limit of the AUC advantage of the ML model to be below the margin; non-inferiority of ML requires the lower limit of its difference from LR to be above minus the margin. Margins were chosen post hoc.")

H2("3.5 Calibration and Recalibration")
cal_rows = [["Setting", "LR slope (95% CI)", "LR CITL (95% CI)", "ML slope, range", "ML CITL, range", "LR Brier: raw / R1 / R2"]]
for s in SET:
    r = row(s, "LR"); z = d[(d.scenario == s) & d.model.isin(ML)]
    cal_rows.append([LAB[s], fci(r.slope, r.slope_lo, r.slope_hi, 2), fci(r.citl, r.citl_lo, r.citl_hi, 2),
                     f"{f(z.slope.min(),2)} to {f(z.slope.max(),2)}", f"{f(z.citl.min(),2)} to {f(z.citl.max(),2)}",
                     f"{f(r.brier)} / {f(r.brier_r1)} / {f(r.brier_r2)}"])
b1 = row("B1_UCI_to_Fram", "LR"); c_ = row("C_Fram_temporal", "LR"); a_va = row("A_VA", "LR")
P(f"Calibration-in-the-large and the Brier score differed far more between settings than between algorithms (Table 7, Figure 3), whereas calibration slopes varied across algorithms within a setting about as much as across settings. In the temporal test (C) the CITL of the six models ranged from {f(d[(d.scenario=='C_Fram_temporal')&d.model.isin(SIX)].citl.min(),2)} to {f(d[(d.scenario=='C_Fram_temporal')&d.model.isin(SIX)].citl.max(),2)} and the calibration slope of LR was {f(c_.slope,2)}. At the VA site the predicted risks were far too low (LR CITL {f(a_va.citl,2)}), which follows from the 74.5% prevalence of angiographic disease against 36–46% in the training sites. In B1 the predicted risks were far too high (LR CITL {f(b1.citl,2)}), because models trained on a population with about 49% disease prevalence were applied to a population with 9% events. These two settings reflect a change in baseline risk and outcome, which no algorithm trained on the source data can correct without local information.")
assert all((six[six.scenario==s_].brier_r2.max()-six[six.scenario==s_].brier_r2.min())<0.01 for s_ in SET)
P(f"Updating the intercept and slope with labelled data from the target site removed most of the miscalibration. For LR the Brier score fell from {f(a_va.brier)} to {f(a_va.brier_r2)} at VA and from {f(b1.brier)} to {f(b1.brier_r2)} in B1 after intercept and slope recalibration (R2); the corresponding change in C was negligible ({f(c_.brier)} to {f(c_.brier_r2)}) because the models were already close to calibrated. After recalibration the Brier scores of the six models differed by less than 0.01 within each setting, again indicating that the choice of algorithm contributes little once the baseline risk is correct.")
TB("Table 7. Calibration of logistic regression and range across the five ML models", [1700, 1600, 1550, 1300, 1300, 2100], cal_rows,
   "Note: CITL, calibration-in-the-large (positive values mean predictions are too low). R1, intercept recalibration; R2, intercept and slope recalibration, both cross-fitted within the target data. The calibration slope has an ideal value of 1 and the CITL of 0.")
IMG("fig_calib.png", "Calibration plots", "Figure 3. Calibration of predicted risks in quantile bins of predicted risk (five bins at the UCI sites, ten in Framingham) for Cleveland, UCI to Framingham and Framingham exam 3")

H2("3.6 Net Benefit")
nbr = [["Setting", "Thresh.", "Treat all", "Age only", "LR", "ML models, range", "ML minus LR, range (n with 95% CI excluding 0)"]]
n_excl = 0
for s in SET:
    for t in sorted(nb[nb.scenario == s].t.unique()):
        z = nb[(nb.scenario == s) & (nb.t == t) & (nb.kind == "recal")]
        lr = z[z.model == "LR"].iloc[0]; mlz = z[z.model.isin(ML)]; ag_ = z[z.model == "AGE"].iloc[0]
        ex = int(((mlz.d_vs_LR_lo > 0) | (mlz.d_vs_LR_hi < 0)).sum()); n_excl += ex
        nbr.append([LAB[s], f(t, 2), f(lr.nb_all), f(ag_.nb), f(lr.nb), f"{f(mlz.nb.min())} to {f(mlz.nb.max())}",
                    f"{f(mlz.d_vs_LR.min())} to {f(mlz.d_vs_LR.max())} ({ex} of 5)"])
n_nb_cmp = sum(1 for r in nbr[1:]) * 5
lrc = nb[(nb.scenario == "C_Fram_temporal") & (nb.kind == "recal") & (nb.model == "LR")]
lrall = nb[nb.scenario.isin(SET) & (nb.kind == "recal") & (nb.model == "LR")]
assert (lrall.d_vs_all >= -1e-9).all()
n_lr_all = int((lrall.d_vs_all_lo > 0).sum())
mlnb = nb[nb.scenario.isin(SET) & (nb.kind == "recal") & nb.model.isin(ML)]
n_pos = int((mlnb.d_vs_LR_lo > 0).sum()); n_neg = int((mlnb.d_vs_LR_hi < 0).sum())
uc = mlnb[mlnb.scenario.isin(["A_Cleveland", "A_Hungary"])]
n30 = int((uc.t == 0.30).sum()); pos30 = int(((uc.t == 0.30) & (uc.d_vs_LR > 0)).sum()); n50 = int((uc.t == 0.50).sum()); neg50 = int(((uc.t == 0.50) & (uc.d_vs_LR < 0)).sum())
agnb = nb[nb.scenario.isin(SET) & (nb.kind == "recal") & (nb.model == "AGE")]
ag_lo = int((agnb.d_vs_LR_hi < 0).sum())
fr_ = mlnb[mlnb.scenario.str.startswith(("B1", "C"))]
assert fr_.d_vs_LR.abs().max() < 0.01, fr_.d_vs_LR.abs().max()
assert pos30 == n30 and neg50 == n50, (pos30, n30, neg50, n50)
_lst = lambda z: "; ".join(f"{MNAME[r.model]} in {LAB[r.scenario]} at {f(r.t,2)}" for r in z.itertuples())
nb_txt = (f"The comparisons favouring an ML model were: {_lst(mlnb[mlnb.d_vs_LR_lo > 0])}. Those favouring LR were: {_lst(mlnb[mlnb.d_vs_LR_hi < 0])}. These intervals are few, but the point estimates show a threshold-dependent pattern at the UCI sites: {pos30} of {n30} ML-minus-LR differences at a threshold of 0.30 were positive and {neg50} of {n50} at 0.50 were negative, so the ML models were slightly better at the lower threshold and worse at the higher one. In Framingham the differences were within about 0.01. No ML model therefore had a clear net benefit advantage. The age-only model had a lower net benefit than LR, with an interval excluding zero, in {ag_lo} of {len(agnb)} setting-threshold combinations. ")
P(f"Table 8 gives net benefit of recalibrated predictions at the pre-specified thresholds. The net benefit of LR was never below that of treating everyone, but the 95% interval of the difference excluded zero in only {n_lr_all} of {len(lrall)} setting-threshold combinations; at the VA site, where the prevalence of 74.5% makes treating everyone a strong default, LR added little. Across {n_nb_cmp} comparisons of an ML model with LR, {n_pos} had a 95% interval of the net benefit difference that excluded zero in favour of the ML model and {n_neg} in favour of LR. {nb_txt}In the temporal test at a threshold of 0.10 the net benefit of LR was {f(lrc[lrc.t==0.10].nb.iloc[0])} against {f(lrc[lrc.t==0.10].nb_all.iloc[0])} for treating everyone, a difference of about {max(0, round(1000*(lrc[lrc.t==0.10].nb.iloc[0]-lrc[lrc.t==0.10].nb_all.iloc[0])))} true-positive equivalents per 1,000 people. Net benefit is expressed per person; intervals for each model are in the supplementary output.")
TB("Table 8. Net benefit of recalibrated predictions at pre-specified thresholds", [1650, 950, 900, 900, 750, 1900, 2650], nbr,
   "Note: Net benefit is expressed as true positives per person, penalising false positives by the odds of the threshold. ML models are RF, GB, SVM, MLP and XGBoost. Predictions are cross-fitted R2 recalibrated.")
IMG("fig_dca.png", "Decision curves", "Figure 4. Decision curves (recalibrated predictions) in Cleveland, VA and Framingham exam 3")

H2("3.7 Clinical Benchmarks")
_f = frs[frs.model != 'FRS']
sig_frs_txt = ', '.join(MNAME[m] for m in _f[_f.d_hi < 0].model)
nsig_frs_txt = ', '.join(MNAME[m] for m in _f[~(_f.d_hi < 0)].model)
frs_rows = [["Model", "AUC (95% CI)", "AUC difference from the Framingham equation (95% CI)"]]
for m in ["LR", "RF", "GB", "SVM", "MLP", "XGB", "AGE", "LRX", "FRS"]:
    r = frr(m)
    frs_rows.append([{"FRS": "Framingham 2008 equation"}.get(m, MNAME[m]), f"{f(r.auc)} ({f(r.auc_lo)}–{f(r.auc_hi)})",
                     "reference" if m == "FRS" else fci(r.d_vs_ref, r.d_lo, r.d_hi)])
P(f"In the {int(frs_ref.n):,} participants of the temporal test set with all inputs of the published Framingham equation ({int(frs_ref.events)} events), the equation reached an AUC of {f(frs_ref.auc)} (95% CI {f(frs_ref.auc_lo)} to {f(frs_ref.auc_hi)}) without any refitting (Table 9). The point estimate was higher than that of every model, by {f(-frr('LR').d_vs_ref)} for LR and {f(-frr('LRX').d_vs_ref)} for the extended LR (ten predictors but no HDL cholesterol). The 95% interval of the difference excluded zero for {sig_frs_txt} but not for {nsig_frs_txt}. An equation that uses the full set of conventional risk factors, including HDL cholesterol and blood-pressure treatment, was therefore not outperformed by any model in this test, and the gap to the five-predictor models is consistent with the value of the missing predictors. Because the equation targets general cardiovascular disease, not CHD only, and was developed in Framingham Heart Study cohorts that may overlap with these participants, this comparison is a benchmark and not a head-to-head evaluation.")
TB("Table 9. Comparison with the published Framingham equation in the temporal test", [2600, 2700, 4400], frs_rows,
   f"Note: Participants with complete inputs for the equation (n = {int(frs_ref.n):,}, {int(frs_ref.events)} events). The equation was applied without refitting. The AUC difference is the model AUC minus the equation AUC.")

H2("3.8 Subgroup Differences at a Fixed Threshold")
sgr = [["Setting", "Measure", "LR difference (95% CI)", "Range over six models"]]
def sgline(s, k, lab):
    z = sg[(sg.kind == "diff") & (sg.scenario == s) & (sg.group == k) & sg.model.isin(SIX)]
    lr = z[z.model == "LR"].iloc[0]; zz = z[z.est.notna()]
    if pd.isna(lr.est): return [LAB[s], lab, "not evaluable", "not evaluable" if zz.empty else f"{f(zz.est.min(),2)} to {f(zz.est.max(),2)}"]
    return [LAB[s], lab, fci(lr.est, lr.lo, lr.hi, 2), f"{f(zz.est.min(),2)} to {f(zz.est.max(),2)}"]
for s in ["A_Cleveland", "A_Hungary", "B1_UCI_to_Fram", "C0_Fram_disjoint", "C_Fram_temporal"]:
    for k, lab in [("dsens_sex", "Sensitivity, men − women"), ("dspec_sex", "Specificity, men − women"), ("dsens_age", "Sensitivity, younger − older"), ("dspec_age", "Specificity, younger − older")]:
        sgr.append(sgline(s, k, lab))
assert (sgd[(sgd.group=="dspec_sex")&sgd.est.notna()&((sgd.lo>0)|(sgd.hi<0))].est<0).all() and sxs[1]==sxs[0]
assert (sgd[(sgd.group=="dsens_age")&sgd.est.notna()].est<0).all() and (sgd[(sgd.group=="dspec_age")&sgd.est.notna()].est>0).all()
sxp = int(((sgd[(sgd.group=="dsens_sex")&sgd.est.notna()&((sgd.lo>0)|(sgd.hi<0))]).est>0).sum())
assert sxp == sx[1], "all significant sex sensitivity differences expected positive"
P(f"Sensitivity at the fixed threshold was higher in men than in women in the evaluable settings (Table 10): {sx[1]} of {sx[0]} model-setting estimates of the sensitivity difference had a 95% interval excluding zero, all of them positive, and the estimates ranged from {f(sx[2],2)} to {f(sx[3],2)}. Specificity showed the opposite pattern: {sxs[1]} of {sxs[0]} evaluable specificity differences excluded zero, all negative (range {f(sxs[2],2)} to {f(sxs[3],2)}). By age, sensitivity was lower and specificity higher in participants younger than the median ({ag[1]} of {ag[0]} sensitivity and {ags[1]} of {ags[0]} specificity intervals excluded zero; ranges {f(ag[2],2)} to {f(ag[3],2)} and {f(ags[2],2)} to {f(ags[3],2)}). The differences in specificity by sex and in sensitivity and specificity by age were systematic in all algorithms, whereas the difference in sensitivity by sex was significant in {sx[1]} of {sx[0]} estimates only. A likely explanation is that sex and age are strong predictors and the groups have different base rates, so a single threshold selects more men and older patients, but we did not test this explanation. The differences do not identify one algorithm as less fair. At the VA site, in Hungary and for several settings of the age comparison the subgroups had too few events or non-events for an estimate.")
TB("Table 10. Signed differences in sensitivity and specificity between subgroups at the training-derived threshold", [1750, 2500, 2750, 2700], sgr,
   "Note: Groups with fewer than 20 events or non-events were not evaluated. Younger and older refer to the training-set median age. Differences are descriptive and are not adjusted for multiplicity.")

H2("3.9 Sensitivity Analyses")
sv = []
for s, lab in [("C_Fram_temporal", "CHD only (primary)"), ("C_S1_composite", "CHD or death"), ("C_S2_exclDeath", "CHD, non-CHD deaths excluded")]:
    z = d[(d.scenario == s)]
    sv.append([lab, f"{int(z.n.iloc[0]):,} ({int(z.events.iloc[0])})", auc_ci(s, "LR"), f"{f(z[z.model.isin(ML)].auc.min())}–{f(z[z.model.isin(ML)].auc.max())}", f(row(s, "AGE").auc), f(row(s, "LR").citl, 2)])
sv_hdr = [["Outcome definition in setting C", "n (events)", "LR AUC (95% CI)", "ML AUC range", "Age-only AUC", "LR CITL"]]
S1 = d[d.scenario == "C_S1_composite"]; S2 = d[d.scenario == "C_S2_exclDeath"]
def sens_cc(R2, name):
    if not R2 or not os.path.exists(f"{R2}/table_discrimination_calibration.csv"): return None
    z = pd.read_csv(f"{R2}/table_discrimination_calibration.csv"); return z
cc = sens_cc(RCC, "cc"); sw = sens_cc(RSW, "sw")
assert row("C_S1_composite","SVM").dauc_vs_LR_hi < 0 and row("C_S2_exclDeath","SVM").dauc_vs_LR_hi < 0
assert (d[d.scenario.isin(["C_S1_composite","C_S2_exclDeath"]) & d.model.isin(ML)].dauc_vs_LR_lo > 0).sum() == 0
txt = (f"Using the composite outcome of CHD or death raised the AUC of LR to {f(row('C_S1_composite','LR').auc)} and the age-only AUC to {f(row('C_S1_composite','AGE').auc)}, as age predicts death strongly. Excluding non-CHD deaths gave an LR AUC of {f(row('C_S2_exclDeath','LR').auc)}, similar to the primary analysis ({f(row('C_Fram_temporal','LR').auc)}). In both variants the SVM was clearly worse than LR (ΔAUC {f(row('C_S1_composite','SVM').dauc_vs_LR)} and {f(row('C_S2_exclDeath','SVM').dauc_vs_LR)}) and no ML model was significantly better (Table 11). ")
if cc is not None:
    ccm = cc[cc.scenario.isin(SET) & cc.model.isin(ML)]
    ccl = cc[cc.scenario.isin(SET) & (cc.model == "LR")].set_index("scenario")
    chg = max(abs(ccl.loc[x, "auc"] - row(x, "LR").auc) for x in SET)
    txt += (f"Restricting the analysis to complete cases (n = {int(ccl.loc['A_VA','n'])} at the VA site, instead of 200) did not change the comparison of ML with LR: {int((ccm.dauc_vs_LR_lo>0).sum())} of {len(ccm)} ML models were significantly better than LR (LR was significantly better in {int((ccm.dauc_vs_LR_hi<0).sum())}), although the LR AUC changed by up to {f(chg)} in individual settings, mainly at the VA site, where only half of the patients have complete data (Table A4). ")
if sw is not None:
    swa = sw[sw.scenario.str.startswith("A_") & sw.model.isin(ML)]
    swb = sw[(sw.scenario == "B1_UCI_to_Fram") & sw.model.isin(ML)]
    swlr = sw[(sw.scenario == "A_Switzerland") & (sw.model == "LR")].iloc[0]
    txt += (f"Adding the Swiss site (123 patients, {int(swlr.events)} with angiographic disease and almost no cholesterol values) gave no discrimination at that site (LR AUC {f(swlr.auc)}, 95% CI {f(swlr.auc_lo)} to {f(swlr.auc_hi)}) and shifted the AUCs at the other sites by up to a few hundredths. {int((swa.dauc_vs_LR_lo>0).sum())} of {len(swa)} ML comparisons at the four UCI sites and {int((swb.dauc_vs_LR_lo>0).sum())} of {len(swb)} in the transfer to Framingham favoured an ML model significantly (LR was significantly better in {int((swa.dauc_vs_LR_hi<0).sum())} of the {len(swa)} comparisons at the UCI sites), a difference of at most {f(swb.dauc_vs_LR.max())} AUC units (Table A4).")
P(txt)
TB("Table 11. Sensitivity of the temporal test (setting C) to the outcome definition", [2700, 1250, 1950, 1400, 1250, 1150], sv_hdr + sv,
   "Note: CITL, calibration-in-the-large. The composite outcome includes death from any cause within 10 years.")

# ================================================================ 4. Discussion
H1("4. Discussion")
H2("4.1 Principal Findings")
P(f"With five common predictors, none of five ML algorithms was significantly better than LR for discrimination in any of six validation settings, and LR was significantly better in {n_worse} of {n_cmp} paired comparisons (Q1). When variability from training was included, LR was non-inferior to ML at a margin of 0.05 in nearly all cases but at stricter margins only in some settings, so the data are consistent with no difference or with differences of a few hundredths in AUC, but the intervals are wide enough that differences of several hundredths in either direction cannot be excluded at the smaller sites (Q2). Calibration after transfer depended on the setting rather than on the algorithm; simple recalibration removed most of the miscalibration and made the algorithms almost indistinguishable in Brier score (Q3). Net benefit of ML models exceeded that of LR in a few threshold-specific comparisons but not consistently (Q4). Finally, a single threshold produced systematically different specificity by sex and different sensitivity and specificity by age in all algorithms, and mostly different sensitivity by sex (Q5).")
H2("4.2 Comparison With Previous Work")
P("These results agree with systematic reviews that found no consistent advantage of ML over LR for tabular clinical data of moderate size [@christ,@nusin], and with evidence from other fields that flawed validation, including leakage, inflates reported ML performance [@kapoor]. Our findings extend this evidence by using participant-disjoint and temporal test data, by reporting calibration, recalibration and net benefit, and by including benchmarks that show what is gained by adding predictors. The finding that a published equation using established risk factors had a higher point-estimate AUC than all models, with a significant gap to most five-predictor models, suggests that the predictors available matter at least as much as the algorithm, although this comparison is not independent of the data used to derive the equation. It also agrees with the view that calibration is the weakest aspect of prediction models after transfer [@vancal] and that the drop in performance between development and later data is a general phenomenon [@subb,@finlayson].")
H2("4.3 Implications")
P("For analysts working with a small number of routinely recorded predictors, this study suggests that a penalised LR is a reasonable default, that the first check at a new site should be calibration, and that a local update of the intercept is cheap and effective when outcome data are available. A model comparison should report paired differences with intervals, an age-only reference and, when possible, a recognised benchmark. Evidence of fairness-related issues should not be inferred from a single threshold: the sensitivity gaps we found may follow from base-rate differences and need a context-specific judgement, for example about whether sex-specific thresholds are acceptable [@hardt]; fairness failures of deployed health algorithms have been documented elsewhere [@ober].")
H2("4.4 Limitations")
P("The study has important limitations. First, the data are old and small. The UCI data were collected in the 1980s from patients referred for angiography, and the Framingham teaching file comes from a cohort examined in the middle of the twentieth century. Clinical practice, risk factor distributions and treatment have changed, so the results should not be taken as evidence on contemporary populations, and we could not access a contemporary dataset. Second, only setting A involves different sites; settings C0 and C use different participants of the same cohort (C at a later examination) and are therefore weaker tests of transport. The UCI and Framingham outcomes differ (angiographic disease versus 10-year incident CHD), so B1 and B2 are stress tests rather than validations of one clinical question, and we describe the whole design as multi-site and temporal validation rather than as validation in an independent population. Third, the five predictors give modest discrimination and the VA, Hungary and Cleveland samples contain 200–303 patients, so the intervals are wide and differences of several hundredths of AUC between algorithms cannot be excluded; the VA site includes very few women. Fourth, the non-inferiority margins were chosen after the analysis and carry no clinical justification, and the full bootstrap fixes hyper-parameters. Fifth, the temporal test is conditional on survival to examination 3, which selects healthier participants, the participants are about ten years older than in training and the glucose definition is not strictly fasting; sensitivity analyses with composite and survivor-excluded outcomes were consistent but do not remove this selection. Sixth, hyperglycaemia is defined differently across sources. Seventh, we used default or modest tuning grids; more extensive tuning might help ML on larger data. Eighth, the analyses were not adjusted for multiplicity, there are no formal tests of the pooled effect across sites, and the Framingham equation comparison involves a different outcome and a cohort overlapping with the development of the equation. Finally, we did not study model updating over time, fairness beyond sex and age, or clinical impact.")
H2("4.5 Future Work")
P("Future studies should repeat this design on large contemporary electronic health record or registry data with the full set of conventional predictors, use pre-registered equivalence margins derived from clinical considerations, include competing-risk and time-to-event methods, and examine updating strategies and transportability across ethnic groups.")

# ================================================================ 5. Conclusions
H1("5. Conclusions")
P(f"In leakage-controlled multi-site and temporal validation with five common predictors, ML algorithms did not discriminate better than logistic regression at unseen sites or at a later examination, and baseline risk and calibration-in-the-large differed far more between settings than between algorithms. Updating the intercept and slope with local data was effective, and a published risk equation with more predictors had a higher point-estimate AUC than every model. Because the data are small and old and the margins of non-inferiority are post hoc, the results support a cautious default of well-specified, recalibrated logistic regression for small tabular problems, not general conclusions about modern ML or contemporary populations.")
MAIN = K; K = []
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_v3_app.py"), encoding="utf8").read())
json.dump({"main": MAIN, "app": K}, open(OUT, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("blocks", len(MAIN), len(K))
