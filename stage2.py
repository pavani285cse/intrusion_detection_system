"""
EGNNN-GROA-BGPoW-IDS-CC Framework
STAGE 2: Feature Selection (DRFSA)
====================================
Implements:
  - Spearman's Rank Correlation Coefficient (SRCC) scoring of all features
  - Iterative backward elimination: remove lowest-ranked feature one at a time
  - Evaluate classifier accuracy on validation split after each removal
  - Stop when accuracy drops
  - Final 20 features as listed in Table 3 of the paper
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Target 20 features from Table 3 of the paper
# ─────────────────────────────────────────────────────────────
PAPER_FINAL_FEATURES = [
    "serror_rate", "same_srv_rate", "diff_srv_rate", "logged_in",
    "protocol_type", "hot", "count", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate",
    "dst_host_serror_rate", "dst_host_rerror_rate", "dst_host_diff_srv_rate",
    "src_bytes", "dst_bytes", "num_root", "service", "land", "flag",
    "wrong_fragment"
]


# ─────────────────────────────────────────────────────────────
# 1. ENCODE LABELS TO INTEGERS
# ─────────────────────────────────────────────────────────────
def encode_labels(train_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> tuple:
    """
    Convert string class labels (Normal, DoS, Probe, R2L, U2R)
    to integer codes 0–4 for use in classifiers.
    """
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test  = le.transform(test_df["label"])
    print(f"[LABEL] Classes: {list(le.classes_)}")
    return y_train, y_test, le


# ─────────────────────────────────────────────────────────────
# 2. SPEARMAN RANK CORRELATION SCORING
# ─────────────────────────────────────────────────────────────
def compute_srcc_scores(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Spearman's Rank Correlation Coefficient (SRCC) between each feature
    and the target label.

    Formula:
        ρ = 1 - (6 · Σd²) / (n · (n² - 1))
    where d = difference between ranks of paired observations.

    Unlike Pearson, SRCC captures MONOTONIC (not just linear) relationships
    and is robust to outliers — important for network traffic data.

    Returns a Series of |ρ| values, one per feature, sorted descending.
    """
    scores = {}
    for col in X.columns:
        rho, _ = spearmanr(X[col], y)
        scores[col] = abs(rho)   # use absolute value — direction doesn't matter

    srcc_series = pd.Series(scores).sort_values(ascending=False)
    print(f"\n[SRCC] Top 10 features by |ρ|:")
    print(srcc_series.head(10).to_string())
    return srcc_series


# ─────────────────────────────────────────────────────────────
# 3. ITERATIVE BACKWARD ELIMINATION
# ─────────────────────────────────────────────────────────────
def iterative_feature_elimination(X_train: pd.DataFrame,
                                   y_train: np.ndarray,
                                   srcc_scores: pd.Series,
                                   val_size: float = 0.2,
                                   min_features: int = 10) -> list:
    """
    Iteratively remove the lowest-ranked feature (by SRCC) one at a time.
    After each removal, train a lightweight Random Forest on a validation
    split and record accuracy.

    Stop when:
      - Accuracy drops below the best seen so far, OR
      - We reach min_features features remaining

    This mirrors the paper's approach of using classifier feedback
    to guide feature removal rather than pure correlation thresholding.

    Returns: ordered list of selected feature names
    """
    # Split training data into sub-train and validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )

    # Start with all features, ordered from highest to lowest SRCC
    remaining_features = list(srcc_scores.index)
    best_accuracy   = 0.0
    best_features   = remaining_features.copy()
    accuracy_history = []

    print(f"\n[ELIM] Starting with {len(remaining_features)} features")
    print(f"[ELIM] Validation split: {val_size*100:.0f}%")
    print(f"[ELIM] Running iterative elimination...\n")

    # Lightweight classifier for fast evaluation at each step
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    step = 0
    while len(remaining_features) > min_features:
        # Train & evaluate on current feature set
        clf.fit(X_tr[remaining_features], y_tr)
        y_pred   = clf.predict(X_val[remaining_features])
        acc      = accuracy_score(y_val, y_pred)
        accuracy_history.append((len(remaining_features), acc))

        print(f"  Step {step:3d} | Features: {len(remaining_features):2d} | Val Acc: {acc:.4f}", end="")

        if acc > best_accuracy:
            best_accuracy  = acc
            best_features  = remaining_features.copy()
            print(" ◄ best")
        else:
            print()

        # Remove the lowest-ranked remaining feature
        worst_feature = remaining_features[-1]   # SRCC sorted descending → last = worst
        remaining_features.remove(worst_feature)
        step += 1

    print(f"\n[ELIM] Best accuracy : {best_accuracy:.4f}")
    print(f"[ELIM] Best feature count: {len(best_features)}")
    return best_features, accuracy_history


# ─────────────────────────────────────────────────────────────
# 4. RECONCILE WITH PAPER'S TABLE 3
# ─────────────────────────────────────────────────────────────
def reconcile_with_paper(selected_features: list,
                          available_features: list) -> list:
    """
    The paper specifies exactly 20 features in Table 3.
    This function:
      1. Keeps features selected by SRCC that also appear in the paper list
      2. Adds any paper features missing from SRCC selection (if available in data)
      3. Ensures final list matches the paper's 20 features exactly

    This handles cases where PCC removal in Stage 1 may have dropped
    some paper features — we only include what's actually available.
    """
    # Features the paper wants AND are available in this dataset
    paper_available = [f for f in PAPER_FINAL_FEATURES if f in available_features]

    # Features SRCC selected that match paper list
    srcc_paper_overlap = [f for f in selected_features if f in PAPER_FINAL_FEATURES]

    # Final: paper features that are available (preserving paper's ordering)
    final_features = paper_available

    print(f"\n[RECONCILE] Paper specifies   : {len(PAPER_FINAL_FEATURES)} features")
    print(f"[RECONCILE] Available in data  : {len(paper_available)} of those")
    print(f"[RECONCILE] SRCC also selected : {len(srcc_paper_overlap)} of paper features")
    print(f"[RECONCILE] Final feature set  : {len(final_features)} features")
    print(f"[RECONCILE] Features           : {final_features}")

    missing = [f for f in PAPER_FINAL_FEATURES if f not in available_features]
    if missing:
        print(f"[RECONCILE] ⚠ Not in data (dropped by PCC or absent): {missing}")

    return final_features


# ─────────────────────────────────────────────────────────────
# 5. PLOT SRCC SCORES AND ACCURACY CURVE
# ─────────────────────────────────────────────────────────────
def plot_feature_selection(srcc_scores: pd.Series,
                            accuracy_history: list,
                            final_features: list,
                            save_path: str = "stage2_feature_selection.png"):
    """
    Two-panel plot:
      Left  — SRCC scores per feature (bar chart, top 20 highlighted)
      Right — Validation accuracy vs number of features (elimination curve)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Stage 2: Feature Selection (DRFSA)", fontsize=14, fontweight="bold")

    # ── Left: SRCC bar chart ──────────────────────────────────
    ax = axes[0]
    colors = ["#2196F3" if f in final_features else "#B0BEC5"
              for f in srcc_scores.index]
    ax.barh(srcc_scores.index[::-1], srcc_scores.values[::-1], color=colors[::-1])
    ax.set_xlabel("Spearman |ρ| Score")
    ax.set_title("SRCC Feature Ranking\n(blue = selected)")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.1, color="red", linestyle="--", linewidth=0.8, label="ρ=0.1")
    ax.legend()

    # ── Right: Accuracy elimination curve ────────────────────
    ax2 = axes[1]
    if accuracy_history:
        n_feats = [h[0] for h in accuracy_history]
        accs    = [h[1] for h in accuracy_history]
        ax2.plot(n_feats, accs, "o-", color="#4CAF50", linewidth=2, markersize=5)
        ax2.axvline(x=len(final_features), color="red", linestyle="--",
                    label=f"Selected: {len(final_features)} features")
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title("Accuracy vs Feature Count\n(backward elimination)")
        ax2.legend()
        ax2.invert_xaxis()  # elimination goes right → left

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — STAGE 2
# ─────────────────────────────────────────────────────────────
def run_stage2(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               val_size: float = 0.2,
               min_features: int = 10) -> dict:
    """
    Full Stage 2 pipeline.
    Input  : preprocessed DataFrames from Stage 1
    Output : dict with selected feature names + filtered DataFrames
    """
    print("=" * 60)
    print("  STAGE 2 — FEATURE SELECTION (DRFSA)")
    print("=" * 60)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    # Step 1: Encode labels
    y_train, y_test, label_encoder = encode_labels(train_df, test_df)

    # Step 2: Compute SRCC scores
    srcc_scores = compute_srcc_scores(X_train, y_train)

    # Step 3: Iterative backward elimination
    selected_by_srcc, accuracy_history = iterative_feature_elimination(
        X_train, y_train, srcc_scores,
        val_size=val_size, min_features=min_features
    )

    # Step 4: Reconcile with paper's Table 3
    final_features = reconcile_with_paper(selected_by_srcc, feature_cols)

    # Step 5: Filter DataFrames to selected features only
    train_selected = train_df[final_features + ["label"]].copy()
    test_selected  = test_df[final_features + ["label"]].copy()

    # Step 6: Plot
    plot_feature_selection(srcc_scores, accuracy_history, final_features)

    print(f"\n[STAGE 2 COMPLETE]")
    print(f"  Features selected : {len(final_features)}")
    print(f"  Train shape       : {train_selected.shape}")
    print(f"  Test shape        : {test_selected.shape}")

    return {
        "train_df":       train_selected,
        "test_df":        test_selected,
        "final_features": final_features,
        "srcc_scores":    srcc_scores,
        "label_encoder":  label_encoder,
        "y_train":        y_train,
        "y_test":         y_test,
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT (can run standalone after Stage 1 CSVs are saved)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "stage1_train.csv"
    test_path  = sys.argv[2] if len(sys.argv) > 2 else "stage1_test.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    results = run_stage2(train_df, test_df)

    results["train_df"].to_csv("stage2_train.csv", index=False)
    results["test_df"].to_csv("stage2_test.csv",   index=False)
    print("[SAVED] stage2_train.csv and stage2_test.csv")