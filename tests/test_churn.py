import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import business  # noqa: E402
from data import clean, load_raw, split  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())


def test_clean_keeps_new_customers(df):
    assert len(df) == 7043
    assert df["churn"].isin([0, 1]).all()
    assert "TotalCharges" not in df.columns


def test_split_keeps_churn_rate(df):
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.customerID).isdisjoint(test.customerID)
    for part in (train, val, test):
        assert abs(part["churn"].mean() - df["churn"].mean()) < 0.01


def test_profit_counts_offer_cost_for_everyone_targeted():
    y = np.array([1, 0, 0])
    bills = np.array([100.0, 100.0, 100.0])
    # one real churner saved with probability 0.3 of 12 * 100, minus three offers of 50
    assert business.profit(y, [True, True, True], bills) == pytest.approx(0.3 * 1200 - 150)
    assert business.profit(y, [False, False, False], bills) == 0


def test_expected_gain_rule():
    gain = business.expected_gain([0.1, 0.9], [20.0, 100.0])
    assert gain[0] < business.OFFER_COST < gain[1]


def test_model_ranks_obvious_cases():
    import joblib

    path = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
    if not path.exists():
        pytest.skip("train the model first: python src/train.py")
    model = joblib.load(path)
    base = {"gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0}
    risky = pd.DataFrame([{**base, "tenure": 1, "Contract": "Month-to-month", "InternetService": "Fiber optic"}])
    safe = pd.DataFrame([{**base, "tenure": 70, "Contract": "Two year", "InternetService": "DSL",
                          "PaymentMethod": "Bank transfer (automatic)"}])
    cols = ["tenure", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
    assert model.predict_proba(risky[cols])[0, 1] > 0.5 > model.predict_proba(safe[cols])[0, 1]
