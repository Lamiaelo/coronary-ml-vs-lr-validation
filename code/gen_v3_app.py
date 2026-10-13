# executed inside gen_v3.py namespace; builds appendix blocks into K, and writes the abstract
H1("Appendix")
P("The tables below support the main text. All values are produced by the supplementary script from the public data.")
# ---- A1
a1 = [["Setting", "Model", "AUC (95% CI)", "ΔAUC vs LR (95% CI)", "Slope", "CITL", "Internal AUC"]]
for s in SET:
    for m in ["LR", "RF", "GB", "SVM", "MLP", "XGB", "AGE", "LRX"]:
        z = d[(d.scenario == s) & (d.model == m)]
        if z.empty: continue
        r = z.iloc[0]
        a1.append([LAB[s], MNAME[m], f"{f(r.auc)} ({f(r.auc_lo)}–{f(r.auc_hi)})", "–" if m == "LR" else fci(r.dauc_vs_LR, r.dauc_vs_LR_lo, r.dauc_vs_LR_hi),
                   f(r.slope, 2), f(r.citl, 2), f(r.internal_cv_auc)])
TB("Table A1. Full discrimination and calibration results", [1450, 900, 1800, 2300, 700, 700, 1000], a1,
   "Note: Internal AUC is the five-fold cross-validated AUC in the training data. Slope, calibration slope; CITL, calibration-in-the-large.")
# ---- A2 (B2)
a2 = [["Test site (trained on Framingham exam 1)", "LR AUC (95% CI)", "ML AUC range", "Age-only AUC", "LR CITL"]]
for site in ["Cleveland", "Hungary", "VA"]:
    z = d[(d.scenario == "B2_Fram_to_UCI") & (d.test == site)]
    if z.empty: continue
    lr = z[z.model == "LR"].iloc[0]; mz = z[z.model.isin(ML)]
    a2.append([site, f"{f(lr.auc)} ({f(lr.auc_lo)}–{f(lr.auc_hi)})", f"{f(mz.auc.min())}–{f(mz.auc.max())}", f(z[z.model == "AGE"].auc.iloc[0]), f(lr.citl, 2)])
P("Setting B2 trains on Framingham examination 1 (10-year incident CHD, 9% prevalence) and tests on each UCI site (angiographic disease, 36–75% prevalence). Discrimination was similar to setting A, whereas predicted risks were far too low because of the difference in prevalence (Table A2).")
TB("Table A2. Cross-outcome transfer from Framingham to the UCI sites (setting B2)", [3000, 2200, 1700, 1500, 1300], a2,
   "Note: CITL values are large and positive because predicted probabilities are too low in the high-prevalence target sites.")
# ---- A3 SHAP
feats = ["age", "sex", "chol", "sbp", "hyperglycaemia"]
a3 = [["Setting", "Age", "Sex", "Cholesterol", "Systolic BP", "Hyperglycaemia"]]
for s in ["A_Cleveland", "A_Hungary", "A_VA", "C_Fram_temporal"]:
    z = sh[sh.scenario == s]
    if z.empty: continue
    a3.append([LAB[s]] + [f(z[c].mean(), 3) for c in feats])
P("Descriptive SHAP values [@lundberg17] show which inputs contributed most to predictions on the test data. Age and sex were the largest contributors in the UCI sites and age was the largest in the temporal test; cholesterol and blood pressure contributed less, and hyperglycaemia contributed little except in Cleveland. The values describe each fitted model and are not tested for stability across resamples (Table A3).")
TB("Table A3. Mean absolute SHAP value per predictor (average over the six models)", [2200, 1100, 1100, 1500, 1500, 2000], a3,
   "Note: Values are in units of the explained model output; compare predictors within a row only.")
# ---- A4 sensitivity
a4 = [["Setting", "Analysis", "LR AUC", "ML AUC range", "ML better than LR (95% CI > 0)", "LR better than ML (95% CI < 0)"]]
def addrows(Rdir, lab, only=None):
    if not Rdir or not os.path.exists(f"{Rdir}/table_discrimination_calibration.csv"): return
    z0 = pd.read_csv(f"{Rdir}/table_discrimination_calibration.csv")
    for s in ["A_Cleveland", "A_Hungary", "A_VA", "A_Switzerland", "B1_UCI_to_Fram", "C0_Fram_disjoint", "C_Fram_temporal"]:
        if only and s not in only: continue
        z = z0[z0.scenario == s]
        if z.empty: continue
        lr = z[z.model == "LR"].iloc[0]; mz = z[z.model.isin(ML)]
        a4.append([LAB.get(s, s.replace("A_", "")), lab, f(lr.auc), f"{f(mz.auc.min())}–{f(mz.auc.max())}", f"{int((mz.dauc_vs_LR_lo > 0).sum())} of 5", f"{int((mz.dauc_vs_LR_hi < 0).sum())} of 5"])
for s in ["A_Cleveland", "A_Hungary", "A_VA", "B1_UCI_to_Fram", "C0_Fram_disjoint", "C_Fram_temporal"]:
    lr = row(s, "LR"); mz = d[(d.scenario == s) & d.model.isin(ML)]
    a4.append([LAB[s], "Primary", f(lr.auc), f"{f(mz.auc.min())}–{f(mz.auc.max())}", f"{int((mz.dauc_vs_LR_lo > 0).sum())} of 5", f"{int((mz.dauc_vs_LR_hi < 0).sum())} of 5"])
addrows(RCC, "Complete cases"); addrows(RSW, "With Swiss site", only=["A_Cleveland", "A_Hungary", "A_VA", "A_Switzerland", "B1_UCI_to_Fram"])
P("Table A4 repeats the main discrimination results for complete cases only and with the Swiss site added to the UCI sites. Swiss cholesterol values are almost entirely recorded as zero (missing), so that site is analysed with imputed cholesterol.")
TB("Table A4. Sensitivity to missing-data handling and to the Swiss site", [1700, 1700, 800, 1500, 2000, 2000], a4,
   "Note: In the analysis with the Swiss site, Switzerland is added to the training data of the other sites and appears as a fourth held-out site. Only the settings that depend on the UCI data are shown for that analysis; the Framingham-trained settings (C0, C) are unchanged by construction.")
# ---- A5 attrition
a5 = [["Included at exam 3", "n", "Age", "Male, %", "Chol", "SBP", "Hyperglyc., %", "Smoker, %", "CHD10, %", "Death10, %"]]
for _, r in att.iterrows():
    a5.append(["Yes" if r.attended_exam3 == 1 else "No", f"{int(r.n):,}", f"{r.age:.1f}", f"{100*r.male:.1f}", f"{r.chol:.1f}", f"{r.sbp:.1f}", f"{100*r.hyperglycaemia:.1f}", f"{100*r.smoker:.1f}", f"{100*r.chd10_exam1:.1f}", f"{100*r.death10_exam1:.1f}"])
P("Participants not included at examination 3 had higher blood pressure, glucose and smoking prevalence at examination 1 (Table A5). Their CHD and death proportions are high by construction, because inclusion at examination 3 requires being alive and CHD-free. The temporal test is therefore a test in survivors who returned, and its event rate is not a population estimate.")
TB("Table A5. Examination 1 characteristics by inclusion at examination 3", [1300, 700, 700, 800, 800, 800, 1100, 1000, 1000, 1100], a5,
   "Note: Means at examination 1; inclusion requires being alive, attending and being CHD-free at examination 3. CHD10 and Death10 are the 10-year CHD and death proportions after examination 1. Chol, cholesterol (mg/dL); SBP, systolic blood pressure (mmHg).")
# ---- A6 PROBAST
a6 = [["Domain", "Risk of bias", "Reason"],
      ["Participants", "High", "Convenience public datasets of referred patients and of one community cohort; small samples; selection into examination 3."],
      ["Predictors", "Low to unclear", "Routine predictors measured before the outcome; hyperglycaemia definition differs between sources."],
      ["Outcome", "High", "Angiographic disease and 10-year incident CHD differ; outcome definition differs across UCI sites; competing deaths addressed only by sensitivity analyses."],
      ["Analysis", "Unclear", "All preprocessing and tuning restricted to training data; participant-disjoint test; small events per site; post hoc margins; no multiplicity control."],
      ["Applicability", "High concern", "Old data, narrow predictor set and different populations from any modern target population."]]
P("We assessed the study with the PROBAST criteria [@probast] (Table A6). The assessment is made by the authors and is therefore subject to bias.")
TB("Table A6. PROBAST self-assessment", [1500, 1500, 6700], a6)
# ---- A7 TRIPOD+AI map
a7 = [["Topic", "Where reported"],
      ["Title, abstract, background and objectives", "Title, Abstract, Section 1"],
      ["Data sources, participants, setting", "Sections 2.1 and 2.3, Tables 1 and 2"],
      ["Outcome and predictors", "Sections 2.1 and 2.2"],
      ["Sample size", "Not calculated; all available data were used (Section 4.4)"],
      ["Missing data", "Section 2.2; Table A4"],
      ["Model building and tuning", "Section 2.4; supplementary code"],
      ["Performance and uncertainty", "Sections 2.5 and 2.6; Section 3"],
      ["Fairness / subgroups", "Section 2.5; Section 3.8"],
      ["Limitations and interpretation", "Section 4.4"],
      ["Code and data availability", "Data Availability Statement"],
      ["Patient and public involvement", "None"]]
TB("Table A7. Map of the TRIPOD+AI reporting items to the manuscript", [4200, 5500], a7, "Note: The checklist is only partly applicable because no new model is proposed for use.")

# ---- abstract (written to file)
n_all5 = sum(nic(sc, "LR_noninf_5") == 5 for sc in ["A_Cleveland", "A_Hungary", "A_VA", "C0_Fram_disjoint", "C_Fram_temporal"])
lrni = {mg: sum(nic(sc, f"LR_noninf_{mg}") == 5 for sc in ["A_Cleveland", "A_Hungary", "A_VA", "C0_Fram_disjoint", "C_Fram_temporal"]) for mg in (2, 3, 5)}
_f = frs[frs.model != "FRS"]
ABS = (f"Machine learning (ML) models for coronary risk are often reported to outperform logistic regression (LR), but evidence that they do so at new sites and later periods, with calibrated predictions and clinical benefit, is limited. "
       f"This study compared LR with five ML algorithms (random forest, gradient boosting, support vector machine, multilayer perceptron and XGBoost) for coronary risk prediction from five common predictors (age, sex, cholesterol, systolic blood pressure and hyperglycaemia) under leakage-controlled validation. "
       f"We used three UCI Heart Disease sites (leave-one-site-out, angiographic disease) and the Framingham teaching dataset (10-year incident coronary heart disease; development and test sets of different participants, the test at the same or at a later examination), plus cross-outcome transfer. "
       f"We report the area under the curve (AUC) with paired bootstrap intervals, non-inferiority from a full bootstrap including training variability (post hoc margins), calibration and recalibration, decision-curve net benefit, subgroup sensitivity, and benchmarks (age-only model, extended LR, published Framingham equation). "
       f"AUCs ranged from {f(auc_min,2)} to {f(auc_max,2)}. No ML model was significantly better than LR in any of {n_cmp} paired comparisons, and LR was significantly better in {n_worse}. "
       f"LR was non-inferior to all five ML models in {lrni[5]} of 5 settings at a margin of 0.05, in {lrni[3]} at 0.03 and in {lrni[2]} at 0.02; intervals were wide, so differences of several hundredths cannot be excluded at the smaller sites. "
       f"Calibration-in-the-large differed far more between settings than between algorithms, and local recalibration removed most miscalibration. ML net benefit did not consistently exceed that of LR. "
       f"The published Framingham equation (AUC {f(frs_ref.auc,2)}; uses HDL cholesterol, treatment and smoking) had a higher point-estimate AUC than every model, significantly so for {int((_f.d_hi < 0).sum())} of {len(_f)}. One threshold gave systematically different specificity by sex and different sensitivity and specificity by age. "
       f"With five common predictors in small, older public datasets, ML showed no clear advantage over LR; baseline risk differed more between settings than the algorithms did. "
       f"The study offers a reproducible template for comparing prediction models beyond internal AUC and states the limits of small, old data.")
open(OUT.replace(".json", "_abstract.txt"), "w", encoding="utf8").write(ABS)
print("abstract words:", len(ABS.split()))
