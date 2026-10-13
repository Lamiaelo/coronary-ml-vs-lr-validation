# Machine learning vs logistic regression for coronary risk: code and results

Code and result files for the article *Multi-Site and Temporal Validation of Machine Learning Versus Logistic Regression for Coronary Risk Prediction With Five Common Predictors: Discrimination, Calibration and Net Benefit* (author: Lamiae El Outouate, FSTT, Abdelmalek Essaadi University, Tetouan, Morocco; lamiae.eloutouate@uae.ac.ma).

## Layout
- `code/run_protocol_v3.py`: runs the full analysis and writes the result tables (CSV).
- `code/make_tables_v3.py`, `stats_v3.py`, `gen_v3.py`, `gen_v3_app.py`: produce figures and the manuscript text/tables from the CSV files.
- `results/main`, `results/sens_complete_case`, `results/sens_swiss`: result files behind the article.
- `figures/`: figures of the article.
- `data/`: instructions for obtaining the public input data (not redistributed).

## Reproduce
```bash
pip install -r requirements.txt
python code/run_protocol_v3.py --uci_dir data/uci --framingham data/frmgham2.csv --seed 2026 --out results/main        --stage all  --B 1000 --Bfull 300
python code/run_protocol_v3.py --uci_dir data/uci --framingham data/frmgham2.csv --seed 2026 --out results/sens_complete_case --stage main --B 300 --complete_case --no_shap --no_variants
python code/run_protocol_v3.py --uci_dir data/uci --framingham data/frmgham2.csv --seed 2026 --out results/sens_swiss --stage main --B 300 --include_swiss --no_shap --no_variants
```
Environment used: Python 3.13, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.52.0, pandas 3.0.5, numpy 2.5.3. About 1 hour on 2 CPU cores. Quick smoke test: add `--fast --B 20 --Bfull 3 --no_shap`.

`gen_v3.py` and `make_tables_v3.py` take result directories as arguments and expect the helper paths used in the original working directory; adjust paths if you rerun them.

## Notes
Public data are old (UCI, 1980s) and small; results are not for clinical use. See the article for limitations.

## Licence
MIT (code). Input data remain under their original terms.
