"""
qsvm_pca_pipeline.py
=====================
QSVM for medication-non-adherence-ml, using the repo's EXISTING
pca_train.csv / pca_test.csv (data/processed/) instead of re-doing PCA.

Your pca_train.csv / pca_test.csv already contain:
    PC1, PC2, ..., PC19, Adherence   (or whatever your target column is called)

Since PCA orders components by explained variance, PC1 is the most
informative, PC2 second most, etc. So instead of re-running PCA, we just
take the FIRST n_qubits columns (PC1..PCn) as the quantum features. This
keeps you on the exact same train/test split your classical models use,
which makes the QSVM-vs-classical comparison as fair as possible.

USAGE
-----
pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn pandas numpy

python qsvm_pca_pipeline.py \
    --train data/processed/pca_train.csv \
    --test  data/processed/pca_test.csv \
    --target Adherence \
    --n_qubits 4 \
    --n_train_samples 150

NOTE ON n_qubits vs 19 PCs
---------------------------
Using all 19 PCs would need 19 qubits, which is very slow to simulate
classically. Taking the first 4-6 PCs is standard practice for quantum
kernels on this kind of tabular data, and PC1-PC4/6 typically already
capture most of the variance (check the print-out — it'll tell you what
fraction of your ORIGINAL pca_train.csv's total variance those columns
correspond to, relative to all 19, so you know what you're giving up).
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif

from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# --------------------------------------------------------------------------
# 1. LOAD ALREADY-PCA'd DATA
# --------------------------------------------------------------------------
def load_pca_data(train_csv, test_csv, target_col, n_qubits, n_train_samples=None,
                   select="variance", n_test_samples=None):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(
            f"'{target_col}' not found in {train_csv}. "
            f"Available columns: {train_df.columns.tolist()}"
        )

    pc_cols_all = [c for c in train_df.columns if c.startswith("PC")]
    pc_cols_all_sorted = sorted(pc_cols_all, key=lambda c: int(c.replace("PC", "")))

    if n_qubits > len(pc_cols_all_sorted):
        raise ValueError(f"n_qubits={n_qubits} but only {len(pc_cols_all_sorted)} PCs available.")

    if select == "variance":
        # Original behavior: PC1..PCn. Only a good idea if PC1 also happens
        # to correlate with the target -- often it doesn't.
        pc_cols = pc_cols_all_sorted[:n_qubits]
    elif select == "mutual_info":
        # Rank ALL PCs by how much they actually tell you about the target
        # (mutual information), not by how much feature-variance they
        # explain, and take the top n_qubits. This is usually the fix when
        # accuracy is stuck near the majority-class baseline.
        mi = mutual_info_classif(
            train_df[pc_cols_all_sorted].values, train_df[target_col].values, random_state=42
        )
        ranked = sorted(zip(pc_cols_all_sorted, mi), key=lambda t: -t[1])
        print("Mutual information with target (top 8 shown):")
        for name, score in ranked[:8]:
            print(f"    {name}: {score:.4f}")
        pc_cols = [name for name, _ in ranked[:n_qubits]]
    else:
        raise ValueError("select must be 'variance' or 'mutual_info'")

    print(f"Using {pc_cols} as quantum features (out of {len(pc_cols_all_sorted)} total PCs, select={select}).")

    # Optional: subsample the TRAINING set only (QSVM training is O(n^2)).
    # Test set stays whole since inference scales linearly (n_train x n_test),
    # not quadratically, and you want a real read on generalization.
    if n_train_samples is not None and n_train_samples < len(train_df):
        train_df = train_df.sample(n=n_train_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled training set to {n_train_samples} rows for QSVM prototyping.")

    if n_test_samples is not None and n_test_samples < len(test_df):
        test_df = test_df.sample(n=n_test_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled test set to {n_test_samples} rows for faster tuning.")

    X_train_raw = train_df[pc_cols].values
    X_test_raw = test_df[pc_cols].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Quantum feature maps encode features as rotation angles -> rescale
    # the chosen PCs into [0, 2*pi). Fit the scaler on train only, apply to
    # both, exactly like you'd do with any other preprocessing step.
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class balance (train): {pd.Series(y_train).value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 2. QUANTUM FEATURE MAP + KERNEL  (same as before)
# --------------------------------------------------------------------------
def build_quantum_kernel(n_qubits, feature_map_type="zz", reps=2):
    if feature_map_type == "z":
        feature_map = z_feature_map(feature_dimension=n_qubits, reps=reps)
    elif feature_map_type == "zz":
        feature_map = zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement="linear")
    else:
        raise ValueError("feature_map_type must be 'z' or 'zz'")
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return kernel, feature_map


# --------------------------------------------------------------------------
# 3. TRAIN + EVALUATE
# --------------------------------------------------------------------------
def run_qsvm(X_train, X_test, y_train, y_test, n_qubits, feature_map_type="zz", reps=2, C=1.0):
    kernel, feature_map = build_quantum_kernel(n_qubits, feature_map_type, reps)

    print(f"\nFeature map: {feature_map_type.upper()}FeatureMap, {n_qubits} qubits, {reps} reps")
    print(f"Training QSVC on {len(X_train)} samples "
          f"(kernel matrix: {len(X_train)}x{len(X_train)} fidelity evaluations)...")

    t0 = time.time()
    qsvc = QSVC(quantum_kernel=kernel, C=C)
    qsvc.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = qsvc.predict(X_test)
    infer_time = time.time() - t0

    metrics = {
        "model": f"QSVM ({feature_map_type.upper()}FeatureMap, reps={reps}, {n_qubits}q)",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 2),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, qsvc.decision_function(X_test))
    except Exception:
        pass

    print("\n--- QSVM results ---")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return qsvc, metrics


def run_classical_baseline(X_train, X_test, y_train, y_test, kernel="rbf", C=1.0):
    # class_weight="balanced" matters here: with a ~55/45 split, an
    # unweighted SVM can just learn to mostly predict the majority class
    # and still look "accurate" while recall on the minority class collapses
    # (which is exactly what your confusion matrix showed).
    svc = SVC(kernel=kernel, C=C, probability=True, class_weight="balanced")
    t0 = time.time()
    svc.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = svc.predict(X_test)

    metrics = {
        "model": f"Classical SVM ({kernel})",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "train_time_s": round(train_time, 2),
        "infer_time_s": None,
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, svc.predict_proba(X_test)[:, 1])
    except Exception:
        pass

    print("\n--- Classical SVM baseline results (same PCs, same split) ---")
    print(classification_report(y_test, y_pred))
    return svc, metrics


# --------------------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QSVM using existing PCA train/test splits")
    parser.add_argument("--train", type=str, default="data/processed/pca_train.csv")
    parser.add_argument("--test", type=str, default="data/processed/pca_test.csv")
    parser.add_argument("--target", type=str, default="Adherence",
                         help="Target column name — confirm the exact spelling in your CSV")
    parser.add_argument("--n_qubits", type=int, default=4, help="How many PCs to use (keep <=6-8)")
    parser.add_argument("--select", type=str, default="mutual_info", choices=["variance", "mutual_info"],
                         help="'mutual_info' picks the PCs most correlated with the target; "
                              "'variance' just takes PC1..PCn (old default, often weaker)")
    parser.add_argument("--n_train_samples", type=int, default=150,
                         help="Subsample training rows for QSVM prototyping (O(n^2) cost)")
    parser.add_argument("--n_test_samples", type=int, default=None,
                         help="Subsample test rows too, for fast tuning iterations. "
                              "QSVM inference cost scales with n_train x n_test, so a "
                              "full 1000-row test set makes every tuning run painfully slow. "
                              "Use e.g. --n_test_samples 100 while tuning, then drop it "
                              "(or set it high) for your final reported numbers.")
    parser.add_argument("--feature_map", type=str, default="zz", choices=["z", "zz"])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_pca_data(
        args.train, args.test, args.target, args.n_qubits, args.n_train_samples,
        args.select, args.n_test_samples
    )

    _, q_metrics = run_qsvm(X_train, X_test, y_train, y_test, args.n_qubits, args.feature_map, args.reps, args.C)
    _, c_metrics = run_classical_baseline(X_train, X_test, y_train, y_test, "rbf", args.C)

    results = pd.DataFrame([q_metrics, c_metrics])
    print("\n=== Summary ===")
    print(results.to_string(index=False))

    from pathlib import Path
    out_path = Path("results/tables/qsvm_pca_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()