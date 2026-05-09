import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv("hw2_train_s4850.csv")
test = pd.read_csv("hw2_test_s4850.csv")

target = "y"
all_predictors = ['x1', 'x2', 'x3']
X_train_raw = train[all_predictors]
y_train = train[target].values
n_train = len(train)
n_test = len(test)

# ── Fit Model to gather stats for the slides ──────────────────────────────
# Creating polynomial/interaction features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X_train_raw), columns=poly.get_feature_names_out())
X_poly_const = sm.add_constant(X_poly)

# Fit Logistic Regression
log_reg = sm.Logit(y_train, X_poly_const).fit(disp=False)
y_pred_prob = log_reg.predict(X_poly_const)
y_pred = (y_pred_prob > 0.5).astype(int)

# Extract params and p-values for the tables
params = log_reg.params.round(3)
p_values = log_reg.pvalues.round(4)

# ══════════════════════════════════════════════════════════════════════════
# CREATE 3-SLIDE PDF (Using the proven Grid Layout)
# ══════════════════════════════════════════════════════════════════════════

with PdfPages("presentation_slides.pdf") as pdf:

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 1: Data
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Data", fontsize=18, fontweight='bold', y=0.98)

    # [MANDATORY 1 & 2] Sample sizes
    info_text = (
        f"Training observations: {n_train}\n"
        f"Test observations: {n_test}\n"
        f"Predictors: {', '.join(all_predictors)}\n"
        f"Response: {target} (binary)"
    )
    axes[0, 0].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', fontfamily='monospace')
    axes[0, 0].set_title("Sample Size & Structure", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    # [MANDATORY 3] Summary statistics
    summary_stats = train.describe().loc[['mean', 'std', 'min', 'max']]
    summary_text = summary_stats.to_string(float_format=lambda x: f"{x:.2f}")
    axes[0, 1].text(0.05, 0.5, summary_text, fontsize=8, verticalalignment='center', fontfamily='monospace')
    axes[0, 1].set_title("Summary Statistics", fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    # [OPTIONAL 1] Correlation heatmap
    corr_matrix = train.corr()
    im = axes[0, 2].imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_xticks(range(len(corr_matrix.columns)))
    axes[0, 2].set_yticks(range(len(corr_matrix.columns)))
    axes[0, 2].set_xticklabels(corr_matrix.columns, fontsize=8)
    axes[0, 2].set_yticklabels(corr_matrix.columns, fontsize=8)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            axes[0, 2].text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    axes[0, 2].set_title("Correlation Heatmap", fontsize=10, fontweight='bold')

    # [MANDATORY 4] Histograms of predictors
    for i, col in enumerate(["x1", "x2", "x3"]):
        ax = axes[1, i]
        ax.hist(train[col].dropna(), bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f"Histogram of {col}", fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 2: Model, Method and Results (Text Only)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Model, Method and Results", fontsize=18, fontweight='bold', y=0.98)

    # [MANDATORY 7] Assumptions
    assump_text = (
        "Model Assumptions:\n"
        "  1. Binary Outcome: y is strictly 0 or 1.\n"
        "  2. Independence: observations are i.i.d.\n"
        "  3. Linearity of Log-Odds: Addressed by\n"
        "     adding degree-2 polynomial features.\n"
        "  4. No severe multicollinearity."
    )
    axes[0, 0].text(0.1, 0.5, assump_text, fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[0, 0].set_title("Assumptions", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # [MANDATORY 6] Library stated
    lib_text = (
        "Method & Libraries:\n"
        "  - Logistic Regression (Logit)\n"
        "  - Polynomial Feature Expansion (deg=2)\n\n"
        "Libraries Used:\n"
        "  - statsmodels.api (Logit modeling)\n"
        "  - scikit-learn (Metrics & preprocessing)\n"
        "  - pandas & numpy (Data handling)\n"
        "  - matplotlib & seaborn (Visualizations)"
    )
    axes[0, 1].text(0.1, 0.5, lib_text, fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[0, 1].set_title("Method & Libraries", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # [MANDATORY 5] Model formula with ALL estimated quantities
    formula_lines = [f"  logit(p) = {params['const']:.3f}"]
    for name in params.index:
        if name != 'const':
            sign = '+' if params[name] > 0 else ''
            formula_lines.append(f"             {sign}{params[name]:.3f}*{name}")
            
    formula_text = "Logistic Regression Formula:\n" + "\n".join(formula_lines)
    
    # Anchored to top (0.95) with smaller font (8) to fit all rows
    axes[1, 0].text(0.05, 0.95, formula_text, fontsize=8, verticalalignment='top', fontfamily='monospace')
    axes[1, 0].set_title("Formula & Quantities", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Result Table with ALL terms
    table_text = f"{'Term':>10s}  {'Coef':>8s}  {'p-value':>8s}\n" + "-" * 30 + "\n"
    for name in params.index:
        table_text += f"{name:>10s}  {params[name]:>8.3f}  {p_values[name]:>8.4f}\n"
    
    # Anchored to top (0.95) with smaller font (8) to fit all rows
    axes[1, 1].text(0.1, 0.95, table_text, fontsize=8, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title("Coefficients Summary", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 3: Visualization
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle("Visualization", fontsize=18, fontweight='bold', y=0.98)

    # [OPTIONAL 2] ROC Curve
    fpr, tpr, _ = roc_curve(y_train, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # [OPTIONAL 3] Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix")

    # [OPTIONAL 4] Predicted vs Actual
    axes[2].scatter(y_train, y_pred_prob, alpha=0.5, color='steelblue')
    axes[2].axhline(y=0.5, color='red', linestyle='--')
    axes[2].set_xticks([0, 1])
    axes[2].set_xlabel("Actual Class (y)")
    axes[2].set_ylabel("Predicted Probability")
    axes[2].set_title("Predicted vs Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print("Saved presentation_slides.pdf (3 slides)")