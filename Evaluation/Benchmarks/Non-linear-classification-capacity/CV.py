import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score, brier_score_loss
from itertools import product

def compute_cv(pathfile, n_fold, limit, n_seed):

    def set_all_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    class TabularDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.norm = nn.BatchNorm1d(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.norm(x)
            x = self.relu(self.fc1(x))
            logits = self.fc2(x)
            proba = torch.softmax(logits, dim=1)
            return logits, proba


    def train_one_split(model, train_loader, num_epochs, lr, device="cpu"):
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()


    @torch.no_grad()
    def eval_split(model, X_eval, y_eval, device="cpu", pos_label=1):
        model.eval()
        X_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
        logits, proba = model(X_t)

        proba = proba.cpu().numpy()
        preds = np.argmax(proba, axis=1)

        f1m = f1_score(y_eval, preds, average="macro")
        pr_auc = average_precision_score(y_eval, proba[:, pos_label])
        br = brier_score_loss(y_eval, proba[:, pos_label])
        return f1m, pr_auc, br



    def make_stable_stratified_folds(ids, y, n_splits=5, seed=50):

        ids = np.asarray(ids)
        y = np.asarray(y)
        fold_id = np.empty(len(y), dtype=int)

        rng = np.random.RandomState(seed)
        classes = np.unique(y)

        for c in classes:
            idx_c = np.where(y == c)[0]
            # shuffle deterministico per classe usando seed
            perm = rng.permutation(idx_c)
            chunks = np.array_split(perm, n_splits)
            for f, chunk in enumerate(chunks):
                fold_id[chunk] = f

        return fold_id


    # ----------------------------
    # 6-fold CV + grid search
    # ----------------------------
    def cv_grid_search(
        X, y,
        ids=None,
        n_splits=3,
        grid=None,
        batch_size_default=64,
        seed=50,
        device="cpu",
        pos_label=1,
        primary_metric="pr_auc",
    ):
        set_all_seeds(seed)

        if ids is None:
            ids = np.arange(len(y))
        ids = np.asarray(ids)

        fold_id = make_stable_stratified_folds(ids, y, n_splits=n_splits, seed=seed)

        if grid is None:
            grid = {
                "hidden_dim": [128, 224, 256],
                "lr": [1e-3, 5e-4, 2e-4],
                "num_epochs": [50, 100],
                "batch_size": [64],
            }

        keys = list(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

        all_rows = []
        best = None

        n_classes = len(np.unique(y))
        input_dim = X.shape[1]

        for ci, params in enumerate(combos, start=1):
            fold_metrics = []

            for f in range(n_splits):
                tr_idx = np.where(fold_id != f)[0]
                te_idx = np.where(fold_id == f)[0]

                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_te, y_te = X[te_idx], y[te_idx]

                train_ds = TabularDataset(X_tr, y_tr)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=params.get("batch_size", batch_size_default),
                    shuffle=True
                )

                set_all_seeds(seed + 1000 * f + 7)

                model = MLP(input_dim=input_dim, hidden_dim=params["hidden_dim"], output_dim=n_classes)
                train_one_split(
                    model,
                    train_loader,
                    num_epochs=params["num_epochs"],
                    lr=params["lr"],
                    device=device
                )

                f1m, prauc, br = eval_split(model, X_te, y_te, device=device, pos_label=pos_label)
                fold_metrics.append((f1m, prauc, br))

            fold_metrics = np.array(fold_metrics)  # (n_splits, 3)

            mean_f1 = fold_metrics[:, 0].mean()
            mean_pr = fold_metrics[:, 1].mean()
            mean_br = fold_metrics[:, 2].mean()
            std_pr  = fold_metrics[:, 1].std()
            std_f1  = fold_metrics[:, 0].std()

            row = {
                **params,
                "cv_f1_macro_mean": mean_f1,
                "cv_pr_auc_mean": mean_pr,
                "cv_pr_auc_std": std_pr,
                "cv_brier_mean": mean_br,
                "cv_f1_std": std_f1,
            }
            all_rows.append(row)

            if primary_metric == "pr_auc":
                score_tuple = (mean_pr, mean_f1, -mean_br)
            elif primary_metric == "f1_macro":
                score_tuple = (mean_f1, mean_pr, -mean_br)
            else:
                raise ValueError("primary_metric deve essere 'pr_auc' o 'f1_macro'")


            if (best is None) or (score_tuple > best["score_tuple"]):
                best = {
                    "params": params,
                    "row": row,
                    "score_tuple": score_tuple,
                    "best_fold_metrics": fold_metrics,  # <---
                }





        results_df = pd.DataFrame(all_rows).sort_values(
            by=("cv_pr_auc_mean" if primary_metric == "pr_auc" else "cv_f1_macro_mean"),
            ascending=False
        ).reset_index(drop=True)


        return results_df, best["params"], fold_id, best["best_fold_metrics"]




    df = pd.read_excel(pathfile)
    X_all = df.iloc[:,0:limit].values
    y_all = df.iloc[:, -1].values


    y_all = y_all
    ids_all = np.arange(len(y_all))


    grid = {
        #"hidden_dim": [64,128, 224, 256, 512],
        "hidden_dim": [256],
        #"hidden_dim": [512, 750, 256, 800],
        "lr": [1e-4],
        #"lr": [1e-3, 1.5e-3, 1e-4, 1.5e-4],
        "num_epochs": [100],
        "batch_size": [64,32]
    }

    results_df, best_params, fold_id, best_fold_metrics = cv_grid_search(
        X_all, y_all,
        ids=ids_all,
        n_splits=n_fold,
        grid=grid,
        seed=n_seed,
        device="cuda",  # "cuda" se disponibile
        pos_label=1,
        primary_metric="f1_macro",
    )

    #print("\n=== BEST PARAMS ===")
    #print(best_params)

    #print("\n=== TOP 10 ===")
    #print(results_df.head(10))

    #print("Best params:", best_params)
    #print("Best per-fold (f1m, pr_auc, brier):\n", best_fold_metrics)

    #print("Mean PR-AUC best:", best_fold_metrics[:,1].mean(), "Std:", best_fold_metrics[:,1].std())

    return best_fold_metrics