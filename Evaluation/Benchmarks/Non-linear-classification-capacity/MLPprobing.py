import os
import glob
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import seaborn as sns
from matplotlib.pyplot import title
from rich import palette
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


from torch.nn.functional import embedding
import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_cv_results(
    matrices,
    model_names=None,
    metric_names=None,
    figsize=(12, 6),
    show_boxplot=True,
    show_heatmap=True
):
    """
    Parameters
    ----------
    matrices : list of np.ndarray or pd.DataFrame
        Ogni matrice ha shape:
        (n_folds, n_metrics)

    model_names : list of str
        Nome di ogni matrice/metodo

    metric_names : list of str
        Nome delle metriche

    """

    n_models = len(matrices)

    colors = [
        "#A63D40",  # Rosso mattone
        "#3A5F8A",  # Blu desaturato
        "#D8A47F",  # Beige caldo
        "#5C7C6F",  # Verde salvia scuro
        "#6C5B7B",  # Viola grigiastro
        "#F4F1EA"  # Panna / Bianco caldo (Nuovo!)
    ]




    matrices = [np.asarray(m) for m in matrices]

    n_metrics = matrices[0].shape[1]

    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(n_models)]

    if metric_names is None:
        metric_names = [f"Metric_{i+1}" for i in range(n_metrics)]


    means = np.array([m.mean(axis=0) for m in matrices])
    stds = np.array([m.std(axis=0) for m in matrices])

    mean_df = pd.DataFrame(
        means,
        index=model_names,
        columns=metric_names
    )

    std_df = pd.DataFrame(
        stds,
        index=model_names,
        columns=metric_names
    )

    print("=== MEAN ===")
    print(mean_df)

    print("\n=== STD ===")
    print(std_df)


    x = np.arange(n_metrics)
    width = 0.8 / n_models

    plt.figure(figsize=figsize)

    for i in range(n_models):
        plt.bar(
            x + i * width,
            means[i],
            width=width,
            yerr=stds[i],
            capsize=5,
            label=model_names[i],
            color = colors[i]
        )

    plt.xticks(x + width * (n_models - 1) / 2, metric_names)
    plt.ylabel("Score")
    plt.title("RESULTS FOR NON-LINEAR CLASSIFICATION CAPACITY")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('grafico1.png')
    print("immagine salvata")


    if show_boxplot:

        rows = []

        for model_name, matrix in zip(model_names, matrices):

            for metric_idx, metric_name in enumerate(metric_names):

                for fold_value in matrix[:, metric_idx]:

                    rows.append({
                        "Model": model_name,
                        "Metric": metric_name,
                        "Value": fold_value
                    })

        df_long = pd.DataFrame(rows)

        plt.figure(figsize=(14, 6))

        sns.boxplot(
            data=df_long,
            x="Metric",
            y="Value",
            hue="Model",
            palette = colors
        )

        plt.title("Distribuzione fold CV")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('grafico2.png')

    # =========================
    # HEATMAP medie
    # =========================
    if show_heatmap:

        plt.figure(figsize=(8, 4))

        sns.heatmap(
            mean_df,
            annot=True,
            cmap="coolwarm",
            fmt=".3f"
        )

        plt.title("Heatmap medie metriche")
        plt.tight_layout()
        plt.savefig('grafico3.png')

    return mean_df, std_df




def plot_custom_style(
        matrices,
        model_names=None,
        metric_names=None,
        figsize=None,
        filename="RepeatedCV_Comparison_2.png"
):

    plt.style.use("seaborn-v0_8-whitegrid")

    n_models = len(matrices)
    matrices = [np.asarray(m) for m in matrices]  # Ogni matrice: (n_repetitions, n_metrics)
    n_metrics = matrices[0].shape[1]


    colors = ["#A63D40", "#3A5F8A", "#D8A47F", "#5C7C6F", "#6C5B7B"]
    colors = colors[:n_models] if n_models <= len(colors) else (colors * int(np.ceil(n_models / len(colors))))[
        :n_models]

    if model_names is None:
        model_names = [f"Model_{i + 1}" for i in range(n_models)]

    if metric_names is None:
        metric_names = [f"Metric_{i + 1}" for i in range(n_metrics)]


    acc_means = np.array([m.mean(axis=0) for m in matrices])  # (n_models, n_metrics)
    acc_stds = np.array([m.std(axis=0) for m in matrices])  # (n_models, n_metrics)


    if figsize is None:
        figsize = (5.5 * n_metrics, 5.0)


    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    np.random.seed(42)


    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]


        for m_idx in range(n_models):
            reps = matrices[m_idx][:, idx]


            jitter = np.random.uniform(-0.12, 0.12, size=len(reps))

            ax.scatter(
                np.ones_like(reps) * m_idx + jitter,
                reps,
                color=colors[m_idx],
                alpha=0.35,
                s=25,
                edgecolors='none',
                zorder=2
            )


        for m_idx in range(n_models):
            ax.errorbar(
                m_idx,
                acc_means[m_idx, idx],
                yerr=acc_stds[m_idx, idx],
                fmt='o',
                color=colors[m_idx],
                ecolor=colors[m_idx],
                elinewidth=2,
                capsize=4,
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=0.8,
                zorder=3
            )


        all_data_this_metric = np.concatenate([m[:, idx] for m in matrices])
        min_val = all_data_this_metric.min()
        max_val = all_data_this_metric.max()


        margin = (max_val - min_val) * 0.15 if max_val != min_val else 0.05
        y_min = max(0.0, min_val - margin)
        y_max = min(1.02, max_val + margin)
        ax.set_ylim(y_min, y_max)


        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8, rotation=15)  # Leggera rotazione se i nomi sono lunghi
        ax.set_title(f"{metric_name} Distribution", fontsize=12, fontweight='bold', pad=10)

        if idx == 0:
            ax.set_ylabel("Metric Value (Mean $\pm$ SD)", fontsize=11)

        ax.grid(axis="y", linestyle="--", alpha=0.4)


        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"Grafico espressivo (Mean + SD) salvato in: {filename}")



RAE =[]
Curia = []
TAP = []
RAE_no_FINE = []
RAE_NO_prep = []
RAE_PREP = []
RAE_SKIP = []
seed = [42, 43 , 44, 45, 30, 47, 48, 39, 33, 32]


# this script perform the Non-linear classification capacity benchmark
import CV
print("performing CV..., 10 repeated 6 fold CV")

for i in range(0,10):
    print(f'iterazione numero{i}')


    best_RAE = CV.compute_cv('../CV_RAE_final.xlsx',n_fold=6, limit=-1, n_seed= seed[i])
    best_RAE = np.mean(best_RAE, axis=0)
    RAE.append(best_RAE)

    best_Curia = CV.compute_cv('CuriaMIP.xlsx',n_fold=6, limit=-1,n_seed= seed[i])
    best_Curia = np.mean(best_Curia, axis=0)
    Curia.append(best_Curia)

    best_TAP = CV.compute_cv('tapMRI.xlsx',n_fold=6, limit=-1, n_seed= seed[i])
    best_TAP = np.mean(best_TAP, axis=0)
    TAP.append(best_TAP)

    baseline_RAE = CV.compute_cv('CV_RAE_BS.xlsx',n_fold=6, limit=512,n_seed= seed[i])
    baseline_RAE = np.mean(baseline_RAE, axis=0)
    RAE_NO_prep.append(baseline_RAE)

    preprocessng_baseline_RAE = CV.compute_cv('../CV_RAE_final_no_prep.xlsx',n_fold=6, limit=-1,n_seed= seed[i])
    preprocessng_baseline_RAE = np.mean(preprocessng_baseline_RAE, axis=0)
    RAE_PREP.append(preprocessng_baseline_RAE)

    skip_baseline_RAE = CV.compute_cv('CV_RAE_Prep_mo_skip.xlsx',n_fold=6, limit=512 ,n_seed= seed[i])
    skip_baseline_RAE = np.mean(skip_baseline_RAE, axis=0)
    RAE_SKIP.append(skip_baseline_RAE)


rae = np.vstack(RAE)
baseline  = np.vstack(RAE_NO_prep)
skip_ablation = np.vstack(RAE_SKIP)
preprocessng_ablation = np.vstack(RAE_PREP)
curiaRes = np.vstack(Curia)
tapRes = np.vstack(TAP)


# GENERATION OF THE PLOTS ----------------------------------------------------------------------------------------------------

means, stds = compare_cv_results(
    [baseline[:,:-1], preprocessng_ablation[:,:-1]],
    model_names=[
        "BASELINE (BS)", "BASELINE (BS) + SKIP CONNECTION (S)",
    ],
    metric_names=[
        "F1-MACRO-AVG",
        "PR-AUC"
 ])


plot_custom_style(
    [baseline[:,:-1], preprocessng_ablation[:,:-1]],
    model_names=[
        "BASELINE (BS)", "BASELINE (BS) + SKIP CONNECTION (S)",
    ],
    metric_names=[
        "F1",
        "PR-AUC"
 ], filename= "grafico_BS_BS_e_SKIP.png")



plot_custom_style(
    [baseline[:,:-1], skip_ablation[:,:-1]],
    model_names=[
        "BASELINE (BS)", "BASELINE (BS) + PREPROCESSING (P)",
    ],
    metric_names=[
        "F1",
        "PR-AUC"
 ], filename= "grafico_BS_BS_e_PREP.png")

#-----------------------------------------------------------------------------------------------------------------------------

means, stds = compare_cv_results(
    [baseline[:,:-1], rae[:,:-1], skip_ablation[:,:-1], preprocessng_ablation[:,:-1],curiaRes[:,:-1], tapRes[:,:-1]],
    model_names=[
        "BASELINE (BS)", "BASELINE+PREPROCESSING+SKIP CONNECTION", "BASELINE+PREPROCESSING", "BASELINE+SKIP CONNECTION","CURIA","TAP-CT-3D"
    ],
    metric_names=[
        "F1",
        "PR-AUC"
 ])










