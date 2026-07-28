import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score

# functions made for the evaluation of the linear probing task

def evaluate_encoder_repeated_cv(X, y, seeds, n_splits=6):
    f1_mean = []
    pr_auc_mean = []
    #f1_scores = []
    #prauc_scores = []

    for seed in seeds:

        f1_scores = []
        prauc_scores = []

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000)
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            f1_scores.append(
                f1_score(
                    y_test,
                    y_pred,
                    average="macro"
                )
            )

            prauc_scores.append(
                average_precision_score(
                    y_test,
                    y_prob
                )
            )

        f1_mean.append(np.mean(f1_scores))
        pr_auc_mean.append(np.mean(prauc_scores))

    return {
        "F1_Macro": np.mean(f1_mean),
        "F1_std": np.std(f1_mean),
        "PR_auc_mean": np.mean(pr_auc_mean),
        "PR_auc_std": np.std(pr_auc_mean),
        "F1_all": f1_mean,
        "PR_auc_all": pr_auc_mean
    }

seeds = [42, 43, 44, 45, 30, 47, 48, 39, 33, 32]

FirstModel = pd.read_excel('CV_finale_Curia.xlsx')
EmbCuria = FirstModel.iloc[:, :-1]

SecondModel = pd.read_excel('CV_RAE_final.xlsx')
EmbRAE = SecondModel.iloc[:, :-1]

SecondModel_noprep = pd.read_excel('CV_RAE_BS.xlsx')
EmbBaseline = SecondModel_noprep.iloc[:, 0:512]

ThirdModel = pd.read_excel('VAL_TAP.xlsx')
EmbTAP= ThirdModel.iloc[:, :-1]

FourthModel = pd.read_excel('CV_RAE_final_no_prep.xlsx')
EmbSkip = FourthModel.iloc[:, :-1]

FifthModel = pd.read_excel('CV_RAE_Prep_mo_skip.xlsx')
EmbPreprocessing = FifthModel.iloc[:, 0:512]


Label = FirstModel.iloc[:, -1]
y = Label

ResultsCuria = evaluate_encoder_repeated_cv(EmbCuria,y,seeds)
ResultsRAE = evaluate_encoder_repeated_cv(EmbRAE,y,seeds)
ResultsTAP = evaluate_encoder_repeated_cv(EmbTAP,y,seeds)
ResultsBaseline = evaluate_encoder_repeated_cv(EmbBaseline,y,seeds)
ResultsSkip = evaluate_encoder_repeated_cv(EmbSkip,y,seeds)
ResultsPrep = evaluate_encoder_repeated_cv(EmbPreprocessing,y,seeds)

names = ["CURIA", "TAP-CT-3D", "BASELINE(BS)", "BASELINE+SKIP CONNECTION(BSK)", "BASELINE+PREPROCESSING(BSP)", "BASELINE+PREPROCESSING+SKIP CONNECTION(BSPA)"]


F1_means = [
    ResultsCuria["F1_Macro"],
    ResultsTAP["F1_Macro"],
    ResultsBaseline["F1_Macro"],
    ResultsSkip["F1_Macro"],
    ResultsPrep["F1_Macro"],
    ResultsRAE["F1_Macro"]
]

F1_stds = [
    ResultsCuria["F1_std"],
    ResultsTAP["F1_std"],
    ResultsBaseline["F1_std"],
    ResultsSkip["F1_std"],
    ResultsPrep["F1_std"],
    ResultsRAE["F1_std"]
]

PR_auc_means = [
    ResultsCuria["PR_auc_mean"],
    ResultsTAP["PR_auc_mean"],
    ResultsBaseline["PR_auc_mean"],
    ResultsSkip["PR_auc_mean"],
    ResultsPrep["PR_auc_mean"],
    ResultsRAE["PR_auc_mean"]
]

PR_auc_stds = [
    ResultsCuria["PR_auc_std"],
    ResultsTAP["PR_auc_std"],
    ResultsBaseline["PR_auc_std"],
    ResultsSkip["PR_auc_std"],
    ResultsPrep["PR_auc_std"],
    ResultsRAE["PR_auc_std"]
]

print("\n=== F1 Macro ===")
for name, mean, std in zip(names, F1_means, F1_stds):
    print(f"{name}: {mean:.4f} ± {std:.4f}")

print("\n=== PR-AUC ===")
for name, mean, std in zip(names, PR_auc_means, PR_auc_stds):
    print(f"{name}: {mean:.4f} ± {std:.4f}")
