from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SEED = 42

# MonthlyCharges is left out of the model: the services explain 99.9% of it (R^2 0.9988), and with both in,
# its coefficient flips sign. Validation AUC is the same without it. It is still used for the offer decision.
NUMERIC = ["tenure"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL, RAW)
    return pd.read_csv(RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is blank for the 11 customers who just joined (tenure 0), so they haven't paid yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    # TotalCharges is basically tenure * MonthlyCharges (correlation 0.9996), so it adds nothing
    return df.drop(columns=["Churn", "TotalCharges"])


def split(df: pd.DataFrame):
    """60/20/20 train/validation/test, same churn rate in each."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df["churn"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["churn"], random_state=SEED)
    return train, val, test


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC + CATEGORICAL]
