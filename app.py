import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ABC Ltd Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load_models():
    return joblib.load("churn_model.joblib"), joblib.load("revenue_model.joblib")

churn, rev = load_models()
clf, THRESH = churn["model"], churn["threshold"]

def yn(x):
    return "Yes" if x else "No"

def reasons_and_actions(c):
    r, a = [], []
    if c["Contract"] == "Month-to-month":
        r.append("On a month-to-month contract (easy to leave)")
        a.append("Offer a discount for moving to a 1 year contract")
    if c["tenure"] <= 6:
        r.append("New customer (6 months or less)")
        a.append("Assign a welcome call and onboarding support")
    if c["InternetService"] == "Fiber optic":
        r.append("Fiber optic plan, which has high churn at ABC Ltd")
        a.append("Check service quality complaints for this customer")
    if c["PaymentMethod"] == "Electronic check":
        r.append("Pays by electronic check")
        a.append("Encourage auto pay (card or bank) with a small reward")
    if c["TechSupport"] == "No" and c["InternetService"] != "No":
        r.append("No tech support subscription")
        a.append("Offer free tech support for 3 months")
    if c["OnlineSecurity"] == "No" and c["InternetService"] != "No":
        r.append("No online security add on")
    if c["MonthlyCharges"] > 80:
        r.append("High monthly bill")
        a.append("Review the plan; suggest a better value bundle")
    return r, a

st.title("📉 ABC Ltd Customer Churn Predictor")
st.caption("Answer a few simple questions about a customer. The tool estimates how likely they are to leave and what you can do about it.")

tab1, tab2, tab3 = st.tabs(["🔍 Check one customer", "📂 Check a customer list", "ℹ️ How it works"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Customer profile")
        tenure = st.slider("How many months has the customer been with us?", 0, 72, 5)
        senior = st.radio("Senior citizen (65+)?", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has a partner/spouse?", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Has dependents (children etc.)?", ["No", "Yes"], horizontal=True)
    with c2:
        st.subheader("Plan and billing")
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.radio("Paperless (online) bill?", ["Yes", "No"], horizontal=True)
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        phone = st.radio("Phone service?", ["Yes", "No"], horizontal=True)
    with c3:
        st.subheader("Add on services")
        no_net = internet == "No"
        lbl = "No internet service"
        sec = lbl if no_net else yn(st.checkbox("Online security", disabled=no_net))
        backup = lbl if no_net else yn(st.checkbox("Online backup", disabled=no_net))
        device = lbl if no_net else yn(st.checkbox("Device protection", disabled=no_net))
        tech = lbl if no_net else yn(st.checkbox("Tech support", disabled=no_net))
        tv = lbl if no_net else yn(st.checkbox("Streaming TV", disabled=no_net))
        movies = lbl if no_net else yn(st.checkbox("Streaming movies", disabled=no_net))
        multi = "No phone service" if phone == "No" else yn(st.checkbox("Multiple phone lines", disabled=phone == "No"))

    reg_row = pd.DataFrame([{"InternetService": internet, "PhoneService": phone, "MultipleLines": multi,
                             "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
                             "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract}])
    est_bill = float(max(rev["model"].predict(reg_row)[0], 18))

    use_est = st.toggle("I know the actual monthly bill", value=False)
    bill = st.number_input("Actual monthly bill ($)", 18.0, 120.0, round(est_bill, 2)) if use_est else est_bill

    if st.button("Predict churn risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": bill, "Contract": contract, "InternetService": internet,
                             "PaymentMethod": payment, "OnlineSecurity": sec, "TechSupport": tech,
                             "PaperlessBilling": paperless, "SeniorCitizen": senior, "Partner": partner,
                             "Dependents": dependents, "StreamingTV": tv}])
        p = float(clf.predict_proba(row)[0, 1])
        if p >= 0.70:
            level, color, msg = "HIGH RISK", "#C44E52", "Act now: contact this customer this week."
        elif p >= THRESH:
            level, color, msg = "MEDIUM RISK", "#DD8452", "Watch closely and consider a retention offer."
        else:
            level, color, msg = "LOW RISK", "#55A868", "Customer is likely to stay. Keep the service good."

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Chance of leaving", f"{p:.0%}")
        m2.metric("Estimated monthly bill", f"${bill:,.2f}")
        m3.metric("Revenue at risk (12 months)", f"${p * bill * 12:,.0f}")
        st.markdown(f"<h3 style='color:{color}'>● {level}</h3><p>{msg}</p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 0.8))
        ax.barh([0], [1], color="#eeeeee"); ax.barh([0], [p], color=color)
        ax.axvline(THRESH, color="black", ls="--", lw=1)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        st.pyplot(fig)

        c = row.iloc[0].to_dict()
        r, a = reasons_and_actions(c)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Why this risk level?")
            for x in (r or ["No major warning signs found"]):
                st.write("• " + x)
        with cc2:
            st.subheader("Suggested actions")
            for x in (a or ["Continue regular engagement"]):
                st.write("✅ " + x)

with tab2:
    st.write("Upload a CSV file of customers (same columns as the ABC Ltd customer file). "
             "You will get a list ranked from highest to lowest risk.")
    up = st.file_uploader("Upload customer CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        if "SeniorCitizen" in data and data["SeniorCitizen"].dtype != object:
            data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
        needed = churn["num"] + churn["cat"]
        missing = [m for m in needed if m not in data.columns]
        if missing:
            st.error(f"These columns are missing: {missing}")
        else:
            data["Churn probability"] = clf.predict_proba(data[needed])[:, 1].round(3)
            data["Risk level"] = pd.cut(data["Churn probability"], [-0.01, THRESH, 0.70, 1.0],
                                        labels=["Low", "Medium", "High"])
            data["Revenue at risk (12m, $)"] = (data["Churn probability"] * data["MonthlyCharges"] * 12).round(0)
            data = data.sort_values("Churn probability", ascending=False)
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers checked", len(data))
            k2.metric("High risk customers", int((data["Risk level"] == "High").sum()))
            k3.metric("Total revenue at risk (12m)", f"${data['Revenue at risk (12m, $)'].sum():,.0f}")
            st.bar_chart(data["Risk level"].value_counts().reindex(["Low", "Medium", "High"]))
            show = ["Churn probability", "Risk level", "Revenue at risk (12m, $)", "tenure", "Contract", "MonthlyCharges"]
            st.dataframe(data[show], use_container_width=True)
            st.download_button("⬇️ Download ranked call list", data.to_csv(index=False), "abc_churn_call_list.csv")

with tab3:
    st.markdown(f"""
**What the tool does:** It learned from the past records of about 7,000 ABC Ltd customers, including who left and who stayed.

**Churn model:** Logistic regression. It gives each customer a chance of leaving (0% to 100%).
Quality score (AUC): **{churn['auc']}** (1.0 is perfect, 0.5 is guessing).

**Bill estimate:** Linear regression based on the services chosen. Average error about **${rev['mae']}** per month.

**Risk levels:** Low below {THRESH:.0%}, Medium between {THRESH:.0%} and 70%, High above 70%.

**Please note:** The tool supports your judgement; it does not replace it.
""")
