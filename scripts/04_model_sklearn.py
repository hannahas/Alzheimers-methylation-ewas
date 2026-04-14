"""
04_model_sklearn.py

Builds a ridge regression model to predict Braak stage
from top CpG methylation values identified in the EWAS.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
FIGURES_DIR = "figures"
EWAS_RESULTS_PATH = os.path.join(DATA_DIR, "ewas_results.parquet")
BETA_PATH = os.path.join(DATA_DIR, "beta_frontal_clean.parquet")
TOP_N_CPGS = 50  # use top 50 CpGs by p-value as features

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 2. Load EWAS results and select top CpGs ──────────────────────────────────
print("Loading EWAS results...")
ewas = pd.read_parquet(EWAS_RESULTS_PATH)
print(f"Total CpGs in results: {len(ewas):,}")

# Sort by p-value, take top N
top_cpgs = (
    ewas
    .dropna(subset=["pvalue"])
    .sort_values("pvalue")
    .head(TOP_N_CPGS)["cpg"]
    .tolist()
)
print(f"\nTop {TOP_N_CPGS} CpGs selected:")
print(ewas[ewas["cpg"].isin(top_cpgs)][["cpg", "slope", "pvalue", "r_squared"]]
      .sort_values("pvalue")
      .to_string(index=False))

# ── 3. Load beta matrix and filter to top CpGs ────────────────────────────────
print(f"\nLoading beta values for top {TOP_N_CPGS} CpGs...")
beta_long = pd.read_parquet(BETA_PATH)
beta_top = beta_long[beta_long["cpg"].isin(top_cpgs)]

# ── 4. Pivot to wide format: rows=samples, cols=CpGs ─────────────────────────
print("Pivoting to wide format...")
beta_wide = beta_top.pivot_table(
    index="barcode",
    columns="cpg",
    values="beta"
)

# Get Braak stage per sample
braak_per_sample = (
    beta_long[["barcode", "braak_stage"]]
    .drop_duplicates()
    .set_index("barcode")
)
beta_wide = beta_wide.join(braak_per_sample)
beta_wide = beta_wide.dropna()

print(f"Feature matrix shape: {beta_wide.shape}")
print(f"Samples: {len(beta_wide)}, Features: {len(top_cpgs)}")

# ── 5. Prepare X and y ────────────────────────────────────────────────────────
X = beta_wide[top_cpgs].values
y = beta_wide["braak_stage"].values

print(f"\nBraak stage distribution in model dataset:")
unique, counts = np.unique(y, return_counts=True)
for stage, count in zip(unique, counts):
    print(f"  Stage {stage:.0f}: {count} samples")

# ── 6. Scale features ─────────────────────────────────────────────────────────
# Ridge regression is sensitive to feature scale so we standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 7. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,       # hold out 20% (16 samples)
    random_state=42
)
print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

# ── 8. Fit RidgeCV ────────────────────────────────────────────────────────────
# RidgeCV automatically finds the best alpha via cross-validation
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)
print(f"\nBest alpha selected by cross-validation: {model.alpha_}")

# ── 9. Evaluate on test set ───────────────────────────────────────────────────
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n=== Model Performance (Test Set) ===")
print(f"R²:                    {r2:.3f}")
print(f"Mean Absolute Error:   {mae:.3f} Braak stages")

# Cross-validated R² on full dataset
cv_scores = cross_val_score(
    RidgeCV(alphas=alphas, cv=5),
    X_scaled, y,
    cv=5,
    scoring="r2"
)
print(f"\nCross-validated R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 10. Predicted vs actual plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1 — predicted vs actual
ax = axes[0]
ax.scatter(y_test, y_pred, color="steelblue", edgecolors="white", s=80, alpha=0.8)
ax.plot([0, 6], [0, 6], color="crimson", linestyle="--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Braak Stage", fontsize=12)
ax.set_ylabel("Predicted Braak Stage", fontsize=12)
ax.set_title("Predicted vs Actual Braak Stage", fontsize=13, fontweight="bold")
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 6.5)
ax.annotate(
    f"R² = {r2:.3f}\nMAE = {mae:.3f}",
    xy=(0.05, 0.85),
    xycoords="axes fraction",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray")
)
ax.legend()

# Plot 2 — feature coefficients (top 20)
ax = axes[1]
coef_df = pd.DataFrame({
    "cpg": top_cpgs,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False).head(20)

colors = ["crimson" if c < 0 else "steelblue" for c in coef_df["coefficient"]]
ax.barh(coef_df["cpg"], coef_df["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Ridge Coefficient", fontsize=12)
ax.set_title("Top 20 CpGs by Coefficient Magnitude", fontsize=13, fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_performance.png"), dpi=150)
plt.show()
print(f"\nFigure saved to {FIGURES_DIR}/model_performance.png")