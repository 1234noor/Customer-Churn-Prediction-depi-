import pickle
import streamlit as st
import pandas as pd
import numpy as np

model = pickle.load(open("customer_churn.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))


st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to churn.")

st.sidebar.header("Customer Information")

credit_score = st.number_input("Credit Score", min_value=0.0)
age = st.number_input("Age", min_value=0.0)
tenure = st.number_input("Tenure", min_value=0.0)
balance = st.number_input("Balance", min_value=0.0)
products_number = st.number_input("Products Number", min_value=1.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

credit_card = st.selectbox("Credit Card", [0, 1])
active_member = st.selectbox("Active Member", [0, 1])

country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])


gender = 1 if gender == "Male" else 0

country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0

# Feature Engineering
balance_salary_ratio = balance / (estimated_salary + 1)

high_balance = 1 if balance > 0.3874024708597822 else 0

balance_log = np.log1p(balance)

active_with_card = 1 if (active_member == 1 and credit_card == 1) else 0


df = pd.DataFrame({

    'credit_score':[credit_score],
    'gender':[gender],
    'age':[age],
    'tenure':[tenure],
    'balance':[balance],
    'products_number':[products_number],
    'credit_card':[credit_card],
    'active_member':[active_member],
    'estimated_salary':[estimated_salary],
    'country_Germany':[country_Germany],
    'country_Spain':[country_Spain],
    'balance_salary_ratio':[balance_salary_ratio],
    'high_balance':[high_balance],
    'balance_log':[balance_log],
    'active_with_card':[active_with_card]

})


num_cols = [
    'credit_score',
    'age',
    'tenure',
    'balance',
    'products_number',
    'estimated_salary'
]

df[num_cols] = scaler.transform(df[num_cols])


# Prediction

if st.button("Predict"):

    result = model.predict(df)
    probability = model.predict_proba(df)

    st.divider()
    st.subheader("Prediction Result")

    if result[0] == 1:

        st.error("Customer Will Churn")

        st.write("### Recommendation")
        st.write("""
- Contact the customer.
- Offer discounts or loyalty rewards.
- Improve customer engagement.
- Follow up with customer support.
""")

        st.progress(100)

    else:

        st.success("Customer Will Not Churn")

        st.write("### Recommendation")
        st.write("""
- Customer is likely to stay.
- Continue providing good service.
- Maintain customer satisfaction.
""")

        st.balloons()

    st.subheader("Prediction Probability")

    st.write(f"**Probability of Not Churn:** {probability[0][0]*100:.2f}%")
    st.write(f"**Probability of Churn:** {probability[0][1]*100:.2f}%")
    
import sklearn
print(sklearn.__version__)
