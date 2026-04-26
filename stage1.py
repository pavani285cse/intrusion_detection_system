"""
EGNNN-GROA-BGPoW-IDS-CC Framework
STAGE 1: Data Loading & Preprocessing (DRFLLS)
================================================
Implements:
  - NSL-KDD loading with column naming
  - Label encoding of categorical features
  - Artificial missing value injection (~5%)
  - Pearson Correlation Coefficient (PCC) redundancy removal (threshold > 0.9)
  - Local Least Squares (LLS) imputation using 7 nearest neighbors
  - Min-Max normalization to [0, 1]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances


# ─────────────────────────────────────────────────────────────
# NSL-KDD column names (41 features + label + difficulty score)
# ─────────────────────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Categorical features to label-encode
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Attack-type grouping (multi-class label)
ATTACK_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "worm": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R", "snmpguess": "U2R",
}


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_nsl_kdd(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD train and test files (space/comma separated .txt).
    Maps raw attack labels to 5-class labels: Normal, DoS, Probe, R2L, U2R.
    Drops the 'difficulty_level' column (not a feature).
    """
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    for df in (train_df, test_df):
        # Normalize label strings (strip trailing dot sometimes present)
        df["label"] = df["label"].str.strip().str.rstrip(".")
        # Map to 5-class attack categories
        df["label"] = df["label"].map(ATTACK_MAP).fillna("Unknown")
        # Drop difficulty score — not used in model
        df.drop(columns=["difficulty_level"], inplace=True, errors="ignore")

    print(f"[LOAD] Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"[LOAD] Train label distribution:\n{train_df['label'].value_counts()}\n")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# 2. LABEL ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────
def encode_categoricals(train_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Label-encode protocol_type, service, flag on train; apply same mapping to test.
    Returns modified DataFrames and the encoder dict for reference.
    """
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        # Fit on union of train+test categories to avoid unseen-label issues
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        encoders[col] = le
        print(f"[ENCODE] '{col}': {len(le.classes_)} unique values → integer codes")

    return train_df, test_df, encoders


# ─────────────────────────────────────────────────────────────
# 3. INJECT MISSING VALUES (~5%)
# ─────────────────────────────────────────────────────────────
def inject_missing_values(df: pd.DataFrame,
                          missing_rate: float = 0.05,
                          seed: int = 42) -> pd.DataFrame:
    """
    Artificially introduce ~5% missing values (NaN) at random positions
    across all feature columns (excluding 'label').
    This simulates real-world incomplete sensor/log data before imputation.
    """
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c != "label"]
    df = df.copy()

    total_cells = df[feature_cols].size
    n_missing   = int(total_cells * missing_rate)

    # Pick random (row, col) indices
    row_idx = rng.integers(0, len(df), size=n_missing)
    col_idx = rng.integers(0, len(feature_cols), size=n_missing)

    for r, c in zip(row_idx, col_idx):
        df.iat[r, c] = np.nan

    actual_pct = df[feature_cols].isna().sum().sum() / total_cells * 100
    print(f"[MISSING] Injected NaNs: {actual_pct:.2f}% of feature cells")
    return df


# ─────────────────────────────────────────────────────────────
# 4. PCC REDUNDANCY REMOVAL (threshold > 0.9)
# ─────────────────────────────────────────────────────────────
def remove_redundant_features_pcc(train_df: pd.DataFrame,
                                   test_df: pd.DataFrame,
                                   threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Pearson Correlation Coefficient (PCC) based redundancy removal.

    Algorithm:
      - Compute pairwise |correlation| matrix on training features
      - For each pair (i, j) where |corr| > threshold:
          remove the feature with the lower mean absolute correlation
          (i.e. keep the more globally correlated / informative one)
      - Drop identified redundant columns from both train and test

    Paper motivation: features that are nearly linearly dependent carry
    duplicate information; removing them reduces dimensionality and
    avoids multicollinearity in downstream models.
    """
    feature_cols = [c for c in train_df.columns if c != "label"]
    # Fill NaN with column median temporarily for correlation computation
    temp = train_df[feature_cols].fillna(train_df[feature_cols].median())

    corr_matrix = temp.corr(method="pearson").abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper_tri.columns:
        # Find all partners with |corr| > threshold
        high_corr_partners = upper_tri.index[upper_tri[col] > threshold].tolist()
        if high_corr_partners:
            # Keep col if it has higher average correlation with all features
            # (proxy for being more "central" / informative)
            col_mean  = corr_matrix[col].mean()
            for partner in high_corr_partners:
                part_mean = corr_matrix[partner].mean()
                # Drop the one with lower mean abs correlation
                if col_mean >= part_mean:
                    to_drop.add(partner)
                else:
                    to_drop.add(col)

    to_drop = list(to_drop)
    train_df = train_df.drop(columns=to_drop, errors="ignore")
    test_df  = test_df.drop(columns=to_drop, errors="ignore")

    remaining = [c for c in train_df.columns if c != "label"]
    print(f"[PCC] Dropped {len(to_drop)} redundant features: {to_drop}")
    print(f"[PCC] Remaining features ({len(remaining)}): {remaining}\n")
    return train_df, test_df, to_drop


# ─────────────────────────────────────────────────────────────
# 5. LOCAL LEAST SQUARES (LLS) IMPUTATION
# ─────────────────────────────────────────────────────────────
def lls_impute(df: pd.DataFrame,
               n_neighbors: int = 7) -> pd.DataFrame:
    """
    Local Least Squares (LLS) imputation using k=7 nearest neighbors.

    For each row with missing values:
      1. Identify observed (non-missing) features in that row → O
      2. Find k nearest complete rows using Euclidean distance on O features
      3. Build:
           K = neighbor values on O features   shape (k, |O|)
           L = neighbor values on missing features shape (k, |M|)
      4. Solve via pseudoinverse (Moore-Penrose):
           x_missing = L^T · (K^T)^†           (from paper Eq.)
         where (K^T)^† = pinv(K^T) = (K·K^T)^{-1}·K
      5. Clip imputed values to [col_min, col_max] of complete rows

    This is superior to mean/median imputation because it preserves
    local feature correlations in the neighborhood.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(float)

    missing_rows = np.where(np.any(np.isnan(X), axis=1))[0]
    complete_rows = np.where(~np.any(np.isnan(X), axis=1))[0]

    print(f"[LLS] Rows with missing values: {len(missing_rows)}")
    print(f"[LLS] Complete rows available for neighbors: {len(complete_rows)}")

    X_complete = X[complete_rows]  # shape (n_complete, n_features)

    for i, row_idx in enumerate(missing_rows):
        row = X[row_idx]
        missing_mask  = np.isnan(row)      # bool mask for missing features
        observed_mask = ~missing_mask      # bool mask for observed features

        observed_indices = np.where(observed_mask)[0]
        missing_indices  = np.where(missing_mask)[0]

        if len(observed_indices) == 0:
            # Entire row missing — fall back to column medians
            for m in missing_indices:
                col_vals = X_complete[:, m]
                X[row_idx, m] = np.nanmedian(col_vals)
            continue

        # ── Step 1: Compute Euclidean distance on observed features only ──
        row_obs = row[observed_indices].reshape(1, -1)        # (1, |O|)
        complete_obs = X_complete[:, observed_indices]        # (n_complete, |O|)

        dists = np.sqrt(np.sum((complete_obs - row_obs) ** 2, axis=1))

        # ── Step 2: Select k nearest neighbors ──────────────────────────
        k = min(n_neighbors, len(complete_rows))
        neighbor_idx = np.argsort(dists)[:k]

        # ── Step 3: Build K and L matrices ──────────────────────────────
        # K: neighbor values on OBSERVED features  → shape (k, |O|)
        K = X_complete[neighbor_idx][:, observed_indices]
        # L: neighbor values on MISSING features   → shape (k, |M|)
        L = X_complete[neighbor_idx][:, missing_indices]

        # ── Step 4: Pseudoinverse solution ──────────────────────────────
        # Paper formula: x = L^T · (K^T)^†
        # (K^T)^† = pinv(K^T)  (shape |O| x k  →  pinv is k x |O|)
        # L^T has shape (|M|, k)
        # Result: (|M|, k) · (k, |O|) = (|M|, |O|) → we need the diagonal?
        # Correct interpretation: estimate missing feature vector x_M
        # by solving K · β = row_obs^T  →  β = pinv(K) · row_obs^T
        # then x_M = L^T · β
        # i.e. x_M = L^T · pinv(K) · row_obs^T
        K_pinv = np.linalg.pinv(K)                           # shape (|O|, k)
        beta   = K_pinv.T @ row_obs.T                        # shape (k, 1)
        x_M    = L.T @ beta                                  # shape (|M|, 1)
        x_M    = x_M.flatten()

        # ── Step 5: Clip to valid range ──────────────────────────────────
        for j, m in enumerate(missing_indices):
            col_min = X_complete[:, m].min()
            col_max = X_complete[:, m].max()
            X[row_idx, m] = np.clip(x_M[j], col_min, col_max)

        if (i + 1) % 1000 == 0:
            print(f"[LLS]   Imputed {i+1}/{len(missing_rows)} rows...")

    df[feature_cols] = X
    remaining_nans = df[feature_cols].isna().sum().sum()
    print(f"[LLS] Imputation complete. Remaining NaNs: {remaining_nans}")
    return df


# ─────────────────────────────────────────────────────────────
# 6. MIN-MAX NORMALIZATION → [0, 1]
# ─────────────────────────────────────────────────────────────
def normalize_features(train_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to all continuous feature columns.
    Scaler is fit ONLY on training data to prevent data leakage.
    Formula: x_norm = (x - x_min) / (x_max - x_min)
    """
    feature_cols = [c for c in train_df.columns if c != "label"]
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    print(f"[NORM] Min-Max normalization applied to {len(feature_cols)} features.")
    print(f"[NORM] Train feature range: [{train_df[feature_cols].min().min():.4f}, "
          f"{train_df[feature_cols].max().max():.4f}]")
    return train_df, test_df, scaler


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — STAGE 1
# ─────────────────────────────────────────────────────────────
def run_stage1(train_path: str = "KDDTrain+.txt",
               test_path:  str = "KDDTest+.txt",
               missing_rate: float = 0.05,
               pcc_threshold: float = 0.9,
               n_neighbors: int = 7) -> dict:
    """
    Full Stage 1 pipeline. Returns a dict with preprocessed DataFrames
    and fitted objects for downstream stages.
    """
    print("=" * 60)
    print("  STAGE 1 — DATA LOADING & PREPROCESSING (DRFLLS)")
    print("=" * 60)

    # Step 1: Load
    train_df, test_df = load_nsl_kdd(train_path, test_path)

    # Step 2: Encode categoricals
    train_df, test_df, encoders = encode_categoricals(train_df, test_df)

    # Step 3: Inject missing values (training set only — simulating real noise)
    train_df = inject_missing_values(train_df, missing_rate=missing_rate)
    test_df  = inject_missing_values(test_df,  missing_rate=missing_rate)

    # Step 4: PCC redundancy removal
    train_df, test_df, dropped_features = remove_redundant_features_pcc(
        train_df, test_df, threshold=pcc_threshold
    )

    # Step 5: LLS imputation
    print("[LLS] Imputing training set...")
    train_df = lls_impute(train_df, n_neighbors=n_neighbors)
    print("[LLS] Imputing test set...")
    test_df  = lls_impute(test_df,  n_neighbors=n_neighbors)

    # Step 6: Normalize
    train_df, test_df, scaler = normalize_features(train_df, test_df)

    print("\n[STAGE 1 COMPLETE]")
    print(f"  Train shape : {train_df.shape}")
    print(f"  Test shape  : {test_df.shape}")
    print(f"  Features    : {[c for c in train_df.columns if c != 'label']}")
    print(f"  Label dist  :\n{train_df['label'].value_counts()}\n")

    return {
        "train_df":        train_df,
        "test_df":         test_df,
        "encoders":        encoders,
        "scaler":          scaler,
        "dropped_features": dropped_features,
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "KDDTrain+.txt"
    test_path  = sys.argv[2] if len(sys.argv) > 2 else "KDDTest+.txt"

    results = run_stage1(train_path=train_path, test_path=test_path)

    # Optionally save preprocessed data for Stage 2
    results["train_df"].to_csv("stage1_train.csv", index=False)
    results["test_df"].to_csv("stage1_test.csv",  index=False)
    print("[SAVED] stage1_train.csv and stage1_test.csv")