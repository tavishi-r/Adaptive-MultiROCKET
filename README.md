# Adaptive MultiROCKET

This project reproduces MultiROCKET for time-series classification and evaluates changes to feature extraction, pooling and downstream classifiers. The improvement experiments focus on **Crop**, with additional checks on **Coffee** and **ECG200**. The notebooks report accuracy, macro F1 and runtime where available.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [MultiRocket_Reproduction.ipynb](MultiRocket_Reproduction.ipynb) | Reproduction of the original implementation, compatibility fixes and benchmark runs on Coffee, ECG200, Adiac, ACSF1 and Crop. |
| [improvements_multiRocekt_final.ipynb](improvements_multiRocekt_final.ipynb) | Feature-budget reduction, feature selection, pooling ablations and alternatives, classifier comparisons, and robustness to noisy and incomplete series. The final section compares candidate pipelines and reports the selection of MultiROCKET with Extra Trees. |

## Run the experiments

You can open either notebook on GitHub to read the saved outputs. To run the improvement experiments locally, clone the repository and launch JupyterLab from its root. On Windows PowerShell:

```powershell
git clone https://github.com/tavishi-r/Adaptive-MultiROCKET.git
cd Adaptive-MultiROCKET
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_improvements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Open the improvements notebook and run the setup cells before the experiment you want to reproduce. It reads the bundled UCR train and test splits from [data/sample](data/sample). The final enhanced-model section requires `multirocket/multirocket_final.py`, which is missing from this checkout. Its saved outputs can still be read in the notebook.

The reproduction notebook uses Google Colab paths and clones the original MultiROCKET repository for its compatibility experiments. Open it in Colab to follow that workflow. It downloads the UCR archive for the broader benchmark and uses Google Drive in its export section.

[requirements.txt](requirements.txt) preserves the original implementation's legacy pins, including a CUDA-specific PyTorch build. [requirements_improvements.txt](requirements_improvements.txt) lists the group notebook dependencies for Python 3.13. CatBoost is included for the optional comparison in Experiment E. The upstream multivariate CLI uses the legacy environment.

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
| **Tavishi Rastogi** | Experimental setup, reproducibility helpers, Crop validation split and baseline. Led feature-budget reduction (A), feature selection (B), pooling ablations and MPV removal (C), and corruption robustness (D). Ran Coffee and ECG200 sanity checks and consolidated the experimental results. |
| **Joviyal** | Evaluated alternative classifiers (E), implemented and evaluated additional pooling operators (F), and compared new-only pooling with the original and combined feature sets (G). |
| **Olena Shvoriak** | Led the final enhanced-model evaluation, comparing feature sets across Ridge, Logistic Regression and Extra Trees over five seeds. Reported the selection and configuration of MultiROCKET-eXtraTrees. |
