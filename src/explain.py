import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data import CATEGORICAL, NUMERIC, ROOT, clean, features, load_raw, split

FIGS = ROOT / "reports" / "figures"
COLUMNS = NUMERIC + CATEGORICAL


def load_model():
    return joblib.load(ROOT / "models" / "churn_model.joblib")


class Explainer:
    """SHAP values (log-odds), added back up per original column
    (the one-hot columns of InternetService become one number for InternetService)."""

    def __init__(self, model, background: pd.DataFrame):
        self.prep = model.named_steps["prep"]
        clf = model.named_steps["model"]
        X_bg = self.prep.transform(features(background))
        if hasattr(clf, "coef_"):
            self.shap = shap.LinearExplainer(clf, X_bg)
        else:
            self.shap = shap.TreeExplainer(clf, shap.sample(X_bg, 200, random_state=0))
        names = [n.split("__", 1)[1] for n in self.prep.get_feature_names_out()]
        self.owner = [next(c for c in COLUMNS if n == c or n.startswith(c + "_")) for n in names]

    def grouped(self, df: pd.DataFrame) -> pd.DataFrame:
        values = np.asarray(self.shap.shap_values(self.prep.transform(features(df))))
        if values.ndim == 3:  # some explainers return one array per class
            values = values[..., 1] if values.shape[-1] == 2 else values[1]
        return pd.DataFrame(values, columns=self.owner).T.groupby(level=0).sum().T.set_index(df.index)

    def reasons(self, customer: pd.DataFrame, n=3):
        row = self.grouped(customer).iloc[0].sort_values(ascending=False)
        return [{"feature": f"{col} = {customer[col].iloc[0]}", "impact": round(float(v), 3)}
                for col, v in row.head(n).items() if v > 0]


def main():
    df = clean(load_raw())
    train, _, test = split(df)
    model = load_model()
    exp = Explainer(model, train)
    sv = exp.grouped(test)

    mean_abs = sv.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_abs.tail(12).plot.barh(ax=ax, color="#3b82f6")
    ax.set_xlabel("Average impact on churn risk (|SHAP|, log-odds)")
    ax.set_title("What drives the predictions (test set)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_impact.png", dpi=120)

    p = model.predict_proba(features(test))[:, 1]
    i = int(np.argsort(p)[-25])
    example = test.iloc[[i]]
    out = {
        "top_impact": mean_abs.tail(8)[::-1].round(3).to_dict(),
        "example_customer": {
            "churn_probability": round(float(p[i]), 3),
            "actually_churned": int(example["churn"].iloc[0]),
            "profile": example[["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod",
                                "TechSupport", "OnlineSecurity"]].iloc[0].to_dict(),
            "top_reasons": exp.reasons(example),
        },
    }
    print(json.dumps(out, indent=2, default=str))
    (ROOT / "reports" / "explanations.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
