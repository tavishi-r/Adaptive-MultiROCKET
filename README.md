# MultiRocket-eXtraTrees: Faster Algorithm For Large Datasets

This AI for Time Series project reproduces the original MultiROCKET baseline and evaluates **MultiRocket-eXtraTrees**, our enhanced model for time-series classification. The model replaces mean positive value (MPV) pooling with temporally partitioned proportion of positive values (TP-PPV) and uses an Extra Trees classifier. We selected the model on Crop in Stage A and benchmarked it against the original MultiROCKET in Stage B on Coffee, ECG200, Adiac, ACSF1 and Crop.

On Crop, MultiRocket-eXtraTrees achieved **76.9% mean accuracy** in **36 seconds median runtime**, compared with **77.6%** and **65 seconds** for the original MultiROCKET with Ridge. That’s about **45% less runtime** with a **0.7 percentage-point drop in accuracy**, suggesting the enhanced model may be useful for larger datasets.

## Environment and data

For the Stage A notebook, launch JupyterLab from the repository root using Python 3.13. In Windows PowerShell, run:

```powershell
git clone https://github.com/tavishi-r/Adaptive-MultiROCKET.git
cd Adaptive-MultiROCKET
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_improvements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

The reproduction report records Python 3.13, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1 and Numba 0.64.0. The group dependency list is unpinned. [requirements.txt](requirements.txt) records the original upstream environment.

Stage A uses the bundled [UCR sample splits](data/sample) for Crop, Coffee and ECG200. The reproduction and Stage B notebooks download Coffee, ECG200, Adiac, ACSF1 and Crop from the UCR dataset archive using `https://www.timeseriesclassification.com/aeon-toolkit/<dataset>.zip`. Run the download cell in each notebook to extract the original train/test splits into `data/UCRArchive_2018/<dataset>/` and convert text files to TSV when needed.

## Reproduce the report

### Original MultiROCKET benchmark (Tables 3–4)

1. Open [MultiRocket_Reproduction.ipynb](MultiRocket_Reproduction.ipynb) in Google Colab in a fresh runtime. Run Steps 1–22 in order to clone the upstream implementation into `/content/MultiRocket` and apply the PyTorch, NumPy and scikit-learn compatibility edits.
2. Run the published-results section and the five-dataset download cell. The metadata inspection references `DataSummary.xlsx` in `/content`. The benchmark uses the downloaded dataset splits and the upstream resampling indices.
3. Under **Reproduction by Benchmark Code**, run the Coffee, ECG200, Adiac, ACSF1 and Crop blocks and then **Reproduction Summary Table**. Use the notebook's 50,000 requested features and one thread. Resample 0 uses the original split, and resamples 1–29 use the upstream indices. Each dataset block writes a `benchmark_summary.csv` and `benchmark_comparison.csv` under `output/multirocket/reproduction/<dataset>/`.

The report compares the authors' published MultiROCKET 50K results with the group's reproduction. Accuracy is mean ± standard deviation over 30 resamples. Times cover training plus testing and were measured on different hardware.

| Dataset | Authors' accuracy (%) | Our accuracy (%) | Authors' time (s) | Our time (s) |
| --- | ---: | ---: | ---: | ---: |
| Coffee | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.60 | 0.42 |
| ECG200 | 89.07 ± 2.41 | 89.27 ± 2.24 | 0.92 | 0.74 |
| Adiac | 82.31 ± 1.56 | 81.96 ± 1.67 | 4.73 | 2.82 |
| ACSF1 | 83.37 ± 4.02 | 83.33 ± 3.92 | 5.78 | 2.78 |
| Crop | 77.39 ± 0.23 | 77.53 ± 0.27 | 85.57 | 106.88 |

### Stage A — Component experiments and model selection

Open [improvements_multiRocekt_final.ipynb](improvements_multiRocekt_final.ipynb) and run sections **0–3** first to set up the five seeds, prepare the Crop validation split and fit the baseline. Then follow the experiment sections below in notebook order.

| Report result | Notebook sections and saved outputs |
| --- | --- |
| Feature budget and selection (Table 5) | Sections **4–5**, including **4.1** for the selected budget on the held-out split. See the feature-budget and selection CSVs in [results/improvements_new](results/improvements_new). |
| Pooling ablation and direct MPV removal (Table 6 and runtime analysis) | Section **6**, experiments **C1–C4**, including the repeated-seed timing and matched-feature-count blocks. |
| Classifier and new-pooling comparisons | Run **E**, **F** and **G** for alternative classifiers, added pooling operators and new-only pooling. See [results/improvements_new](results/improvements_new) and the [new-only results by seed](10_new_only_pooling_by_seed.csv). |
| Corruption robustness and cross-dataset checks | Run **8–8.2** for Experiment D and **9** for Coffee and ECG200. See [Experiment_D_Results](Experiment_D_Results). |
| Final model selection | Section **12** compares two feature sets and three classifiers over five seeds. Run the section after the shared setup. It imports the enhanced transform from [multirocket_extratrees.py](multirocket/multirocket_extratrees.py). |

### Stage B — Final model benchmark (Tables 7–8 and Figures 1–3)

Open [MultiRocket_StageB_improvements.ipynb](MultiRocket_StageB_improvements.ipynb) in a fresh Google Colab runtime. Run **1–4** to install dependencies, copy the upstream source and indices, write the frozen Extra Trees model, download the data and warm up Numba.

- For **Figures 1–2**, keep `K = 49`, `N_ESTIMATORS = 300`, `NUM_FEATURES_ORIG = 50000` and `N_RESAMPLES = 10` in the setup cell. Run **5–8** to evaluate Extra Trees, original MultiROCKET with Ridge, and the logistic-regression control. Section 8 writes `accuracy_comparison.png` and `time_comparison.png`.
- For **Tables 7–8 (Crop, official split)**, run Section **9** and filter `crop_grid.csv` to `TrainSize == 7200` and `Resample == 0`. Use `Accuracy` for Table 7 and `Total_s` (fit plus prediction) for Table 8. The CSV's `NumFeatures` column records nominal feature budgets.
- For **Figure 3 (Crop scaling)**, use Section **9** and average `Accuracy` across resamples for each combination of model, training-set size and feature budget. The default sweep uses three runs. Run 0 uses the official split, and runs 1 and 2 use the authors' resampling indices.

## Source and results

- [multirocket/](multirocket) contains the original transforms and the implemented pooling variants.
- [Experiment_D_Results/](Experiment_D_Results) contains the robustness validation tables, held-out test results and plots.
- [results/](results) contains benchmark and improvement results. [10_new_only_pooling_by_seed.csv](10_new_only_pooling_by_seed.csv) contains the new-only pooling results by seed.

## Attribution and license

Based on [the original MultiRocket implementation](https://github.com/ChangWeiTan/MultiRocket) by Chang Wei Tan, Angus Dempster, Christoph Bergmeir and Geoffrey I. Webb. See [MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification](https://arxiv.org/abs/2102.00457).

This repository is distributed under the [GNU General Public License v3.0](LICENSE).

## Individual contributions

| Team member | Contributions |
| --- | --- |
| **Olena Shvoriak** | Reproduced the original benchmark across five datasets and 30 resamples, applied compatibility fixes, led Stage A model selection and Stage B benchmarking and scaling experiments, and contributed to report writing. |
| **Tavishi Rastogi** | Experimental setup, reproducibility helpers, Crop validation split and baseline. Led feature-budget reduction (A), feature selection (B), pooling ablations and MPV removal (C), and corruption robustness (D). Ran Coffee and ECG200 sanity checks, consolidated experimental results, and prepared the notebooks and GitHub repository. |
| **Joviyal Rose Anandan** | Evaluated alternative classifiers (E), implemented and evaluated additional pooling operators (F), and compared new-only pooling with the original and combined feature sets (G). |
