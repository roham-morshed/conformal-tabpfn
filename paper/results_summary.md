# Results summary

Every number below is read directly from a measured row in `results/raw/*.parquet`. Nothing is interpolated or estimated.

- rows: 130896  |  cells: 14544  |  failures: 0
- datasets: 15  |  models: ['logreg', 'rf', 'tabpfn', 'xgboost']
- cp methods: ['cvplus_K10', 'cvplus_K20', 'cvplus_K5', 'jackknife_plus', 'mondrian_cvplus_K10', 'mondrian_split_0.3', 'pooled_cv_K10', 'split_0.1', 'split_0.15', 'split_0.2', 'split_0.25', 'split_0.3', 'split_0.4', 'split_0.5']
- seeds per cell: 20  |  alphas: [np.float64(0.05), np.float64(0.1), np.float64(0.2)]
- devices used: ['cpu', 'cuda']
- quantile method: ['tight']

**T1** covers 15 datasets x 8 CP methods.
Best mean rank by set size: **cvplus_K20** (2.47); worst: mondrian_split_0.3 (6.47).
Nemenyi critical difference (alpha=0.05, k=8, N=15): 2.71.

### RQ1 -- wall-clock cost
  tabpfn: Split total 0.654 s, of which fit is 22.8%.
  xgboost: Split total 0.061 s, of which fit is 87.3%.
  rf: Split total 0.426 s, of which fit is 90.5%.
  logreg: Split total 0.013 s, of which fit is 60.2%.

### RQ1 -- cost ratio CV+/Split (the thesis test)
- cvplus_K10: tabpfn=11.28x, xgboost=10.09x, rf=10.38x, logreg=10.19x
- cvplus_K20: tabpfn=25.88x, xgboost=19.89x, rf=20.82x, logreg=20.28x
- cvplus_K5: tabpfn=4.91x, xgboost=5.01x, rf=5.06x, logreg=5.05x
- cvplus_K10: TabPFN ratio 11.28x vs gradient-model mean 10.22x -> TabPFN is 1.10x the baseline ratio (NO advantage).
- cvplus_K20: TabPFN ratio 25.88x vs gradient-model mean 20.33x -> TabPFN is 1.27x the baseline ratio (NO advantage).
- cvplus_K5: TabPFN ratio 4.91x vs gradient-model mean 5.04x -> TabPFN is 0.97x the baseline ratio (NO advantage).

### RQ2 -- coverage stability (headline)
- cvplus_K10: median sd(Split)/sd(CV+) = **1.27x**; CV+ has lower sd on 14/15 datasets (Wilcoxon p=0.0009).
- cvplus_K20: median sd(Split)/sd(CV+) = **1.26x**; CV+ has lower sd on 14/15 datasets (Wilcoxon p=0.0001).
- cvplus_K5: median sd(Split)/sd(CV+) = **1.22x**; CV+ has lower sd on 13/15 datasets (Wilcoxon p=0.0026).

### RQ4 -- conservativeness of CV+ (guarantee is 1-2a = 0.80, nominal 1-a = 0.90)
- binary, K=5: coverage 0.9079 (over-coverage +0.0079), set size 1.099
- binary, K=10: coverage 0.9046 (over-coverage +0.0046), set size 1.092
- binary, K=20: coverage 0.9022 (over-coverage +0.0022), set size 1.088
- multiclass, K=5: coverage 0.9093 (over-coverage +0.0093), set size 1.280
- multiclass, K=10: coverage 0.9058 (over-coverage +0.0058), set size 1.268
- multiclass, K=20: coverage 0.9024 (over-coverage +0.0024), set size 1.262

### RQ5 -- class-conditional fairness
- mondrian_split_0.3 vs split_0.3: mean gap 0.0697 vs 0.3364; Mondrian narrower on 18/20 (p=0.0000); set size 1.304 vs 1.104.
- mondrian_cvplus_K10 vs cvplus_K10: mean gap 0.0568 vs 0.3401; Mondrian narrower on 20/20 (p=0.0000); set size 1.217 vs 1.098.

### Alpha robustness
- alpha=0.05 (target 0.95): cvplus_K10=0.953, cvplus_K20=0.951, cvplus_K5=0.956, jackknife_plus=0.947, mondrian_cvplus_K10=0.956, mondrian_split_0.3=0.961, pooled_cv_K10=0.953, split_0.3=0.954
- alpha=0.10 (target 0.90): cvplus_K10=0.905, cvplus_K20=0.902, cvplus_K5=0.908, jackknife_plus=0.898, mondrian_cvplus_K10=0.909, mondrian_split_0.3=0.908, pooled_cv_K10=0.906, split_0.3=0.902
- alpha=0.20 (target 0.80): cvplus_K10=0.806, cvplus_K20=0.804, cvplus_K5=0.810, jackknife_plus=0.817, mondrian_cvplus_K10=0.809, mondrian_split_0.3=0.809, pooled_cv_K10=0.810, split_0.3=0.804

### RQ3 -- calibration budget
- tabpfn: rho* = **0.20** (min set size 1.144, coverage 0.9021, underlying accuracy 0.8705)
- xgboost: rho* = **0.30** (min set size 1.223, coverage 0.9014, underlying accuracy 0.8380)

### RQ6 -- crossover in n_train
- cvplus_K10: n=100 sd=0.0363 size=1.375, n=200 sd=0.0256 size=1.316, n=400 sd=0.0214 size=1.293, n=800 sd=0.0212 size=1.310
- split_0.3: n=100 sd=0.0484 size=1.386, n=200 sd=0.0374 size=1.322, n=400 sd=0.0296 size=1.302, n=800 sd=0.0294 size=1.309
- Split catches CV+ on coverage-sd at n = not within the swept range.
