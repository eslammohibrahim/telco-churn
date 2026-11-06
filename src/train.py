import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import business
from data import CATEGORICAL, NUMERIC, ROOT, SEED, clean, features, load_raw, split

FIGS = ROOT / "reports" / "figures"

MODELS = {
    "Logistic regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10, 100, 1000]}),
    "Random forest": (
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 20, 50, 100]},
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"model__learning_rate": [0.01, 0.03, 0.1], "model__max_depth": [2, 3, 6], "model__max_iter": [50, 100, 300]},
    ),
}


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def scores(y, p):
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
    }


def eda_plots(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    by_contract = df.groupby("Contract")["churn"].mean().reindex(["Month-to-month", "One year", "Two year"])
    tenure_bins = df.assign(t=pd_cut(df["tenure"])).groupby("t", observed=True)["churn"].mean()
    by_internet = df.groupby("InternetService")["churn"].mean().reindex(["No", "DSL", "Fiber optic"])
    for ax, series, title in [(axes[0], by_contract, "Contract"), (axes[1], tenure_bins, "Months as a customer"),
                              (axes[2], by_internet, "Internet service")]:
        ax.bar(series.index.astype(str), series.values, color="#3b82f6")
        ax.set_title(title, loc="left")
        for i, v in enumerate(series.values):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    axes[0].set_ylabel("Churn rate")
    axes[0].set_ylim(0, 0.65)
    fig.tight_layout()
    fig.savefig(FIGS / "churn_rates.png", dpi=120)


def pd_cut(tenure):
    import pandas as pd
    return pd.cut(tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = clean(load_raw())
    train, val, test = split(df)
    print(f"train {len(train)}, val {len(val)}, test {len(test)}, churn rate {df['churn'].mean():.3f}")
    eda_plots(df)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fitted, report = {}, {"models": {}}
    for name, (model, grid) in MODELS.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(features(train), train["churn"])
        p_val = search.predict_proba(features(val))[:, 1]
        fitted[name] = search.best_estimator_
        report["models"][name] = {"best_params": search.best_params_, "cv_roc_auc": round(float(search.best_score_), 4),
                                  "val": scores(val["churn"], p_val)}
        print(f"{name:<20} {report['models'][name]}")

    best = max(report["models"], key=lambda n: report["models"][n]["val"]["roc_auc"])
    model = fitted[best]
    print(f"\npicked on validation: {best}")

    # threshold for the retention offer, chosen on validation
    p_val = model.predict_proba(features(val))[:, 1]
    t_best, val_profit = business.best_threshold(val["churn"], p_val, val["MonthlyCharges"])

    p_test = model.predict_proba(features(test))[:, 1]
    y, mc = test["churn"].to_numpy(), test["MonthlyCharges"].to_numpy()
    expected_gain = business.expected_gain(p_test, mc)
    strategies = {
        "Nobody": np.zeros_like(y, dtype=bool),
        "Everyone": np.ones_like(y, dtype=bool),
        "Probability >= 0.5": p_test >= 0.5,
        f"Probability >= {t_best:.2f} (tuned on validation)": p_test >= t_best,
        "Expected gain > offer cost": expected_gain > business.OFFER_COST,
    }
    report["test"] = scores(y, p_test)
    report["picked_model"] = best
    report["threshold"] = round(float(t_best), 2)
    report["strategies_test"] = {
        k: {"customers_targeted": int(m.sum()), "churners_reached": int(y[m].sum()),
            "profit": round(business.profit(y, m, mc))} for k, m in strategies.items()
    }
    report["assumptions"] = {"offer_cost": business.OFFER_COST, "accept_rate": business.ACCEPT_RATE,
                             "months_kept": business.MONTHS_KEPT}
    print("\ntest:", report["test"])
    for k, v in report["strategies_test"].items():
        print(f"  {k:<48} {v}")

    plot_curves(fitted, val, y, p_test, best)
    plot_profit(y, p_test, mc, t_best)
    joblib.dump(model, ROOT / "models" / "churn_model.joblib")
    (ROOT / "reports" / "results.json").write_text(json.dumps(report, indent=2, default=str))


def plot_curves(fitted, val, y_test, p_test, best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, m in fitted.items():
        p = m.predict_proba(features(val))[:, 1]
        prec, rec, _ = precision_recall_curve(val["churn"], p)
        axes[0].plot(rec, prec, label=f"{name} (AP {average_precision_score(val['churn'], p):.2f})")
    axes[0].axhline(val["churn"].mean(), color="gray", ls="--", lw=1, label="Random")
    axes[0].set_xlabel("Recall (share of churners caught)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-recall, validation set", loc="left")
    axes[0].legend(fontsize=8)

    frac, mean_p = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Perfect")
    axes[1].plot(mean_p, frac, marker="o", label=best)
    axes[1].set_xlabel("Predicted churn probability")
    axes[1].set_ylabel("Actual churn rate")
    axes[1].set_title("Calibration, test set", loc="left")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "model_curves.png", dpi=120)


def plot_profit(y, p, mc, t_best):
    grid = np.linspace(0.05, 0.95, 91)
    profits = [business.profit(y, p >= t, mc) for t in grid]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, profits, color="#16a34a")
    ax.axvline(t_best, color="gray", ls="--", lw=1)
    ax.text(t_best + 0.01, max(profits) * 0.95, f"chosen on validation: {t_best:.2f}", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Send the offer to customers with churn probability above...")
    ax.set_ylabel("Profit on test set ($)")
    ax.set_title("Where to draw the line", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "profit_threshold.png", dpi=120)


if __name__ == "__main__":
    main()
