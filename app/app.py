"""Streamlit dashboard: churn risk for one customer, and who to contact.

Run:  streamlit run app/app.py   (train first: python src/train.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import business  # noqa: E402
from data import clean, features, load_raw, split  # noqa: E402
from explain import Explainer, load_model  # noqa: E402

st.set_page_config(page_title="Churn risk", layout="wide")


@st.cache_resource
def setup():
    df = clean(load_raw())
    train, _, test = split(df)
    model = load_model()
    return df, test, model, Explainer(model, train)


df, test, model, explainer = setup()

st.sidebar.header("Offer assumptions")
offer_cost = st.sidebar.number_input("Cost of one offer ($)", 0.0, 500.0, business.OFFER_COST, 5.0)
accept_rate = st.sidebar.slider("Churners who stay because of the offer", 0.05, 1.0, business.ACCEPT_RATE, 0.05)
months_kept = st.sidebar.slider("Months of revenue kept if they stay", 1, 36, business.MONTHS_KEPT)

st.title("Customer churn risk")
tab_one, tab_list = st.tabs(["One customer", "Who to contact"])

with tab_one:
    def options(col):
        return sorted(df[col].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Months as a customer", 0, 72, 3)
        monthly = st.number_input("Monthly bill ($)", 18.0, 120.0, 85.0, 0.5)
        contract = st.selectbox("Contract", options("Contract"))
        payment = st.selectbox("Payment method", options("PaymentMethod"), index=2)
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with c2:
        internet = st.selectbox("Internet", options("InternetService"), index=1)
        security = st.selectbox("Online security", options("OnlineSecurity"))
        support = st.selectbox("Tech support", options("TechSupport"))
        backup = st.selectbox("Online backup", options("OnlineBackup"))
        protection = st.selectbox("Device protection", options("DeviceProtection"))
    with c3:
        tv = st.selectbox("Streaming TV", options("StreamingTV"))
        movies = st.selectbox("Streaming movies", options("StreamingMovies"))
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", options("MultipleLines"))
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    customer = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "Contract": contract, "PaymentMethod": payment,
        "PaperlessBilling": paperless, "InternetService": internet, "OnlineSecurity": security,
        "TechSupport": support, "OnlineBackup": backup, "DeviceProtection": protection, "StreamingTV": tv,
        "StreamingMovies": movies, "PhoneService": phone, "MultipleLines": lines, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "gender": gender,
    }])
    p = float(model.predict_proba(features(customer))[:, 1][0])
    gain = float(business.expected_gain(p, monthly, accept_rate, months_kept))

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{p:.0%}")
    m2.metric("Expected saving from an offer", f"${gain:,.0f}")
    m3.metric("Decision", "Send offer" if gain > offer_cost else "Don't send", f"offer costs ${offer_cost:,.0f}",
              delta_color="off")

    reasons = explainer.reasons(customer, n=4)
    if reasons:
        st.subheader("Why the risk is high")
        for r in reasons:
            st.write(f"- **{r['feature']}** (+{r['impact']:.2f} log-odds)")

with tab_list:
    p_test = model.predict_proba(features(test))[:, 1]
    gains = business.expected_gain(p_test, test["MonthlyCharges"], accept_rate, months_kept)
    target = gains > offer_cost
    y = test["churn"].to_numpy()
    total = business.profit(y, target, test["MonthlyCharges"], offer_cost, accept_rate, months_kept)

    st.write(f"Held-out test customers: {len(test):,}. With these assumptions the rule says contact "
             f"**{target.sum():,}** of them. Looking at who actually left, that would have reached "
             f"**{y[target].sum():,}** of the {y.sum():,} churners and earned about **${total:,.0f}** "
             f"after paying for the offers.")
    table = test.assign(churn_probability=p_test.round(3), expected_saving=gains.round(0))[target]
    st.dataframe(
        table.sort_values("expected_saving", ascending=False)[
            ["customerID", "churn_probability", "expected_saving", "tenure", "MonthlyCharges", "Contract",
             "InternetService", "churn"]
        ].rename(columns={"churn": "actually_left"}),
        hide_index=True, use_container_width=True,
    )
