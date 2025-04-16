import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- SECTION 0: Preparing data ---
# Loading data
@st.cache_data
def load_data():
    churn_df = joblib.load("data/churn_df.pkl")
    X_test_df = joblib.load("data/X_test_df.pkl")
    best_model = joblib.load("model/best_model.pkl")

    return churn_df, X_test_df, best_model

churn_df, X_test_df, best_model = load_data()

# --- SECTION 1: Sidebar Filters ---
# Creating filters to manually specify which features to include in the dataframe
st.sidebar.header("Filters")
selected_gender = st.sidebar.multiselect("Gender", churn_df["Gender"].unique(), default=churn_df["Gender"].unique())
selected_senior_citizen = st.sidebar.multiselect("Senior Citizen", churn_df["Senior Citizen"].unique(), default=churn_df["Senior Citizen"].unique())
selected_partner = st.sidebar.multiselect("Partner", churn_df["Partner"].unique(), default=churn_df["Partner"].unique())
selected_dependents = st.sidebar.multiselect("Dependents", churn_df["Dependents"].unique(), default=churn_df["Dependents"].unique())
selected_phone_service = st.sidebar.multiselect("Phone Service", churn_df["Phone Service"].unique(), default=churn_df["Phone Service"].unique())
selected_multiple_lines = st.sidebar.multiselect("Multiple Lines", churn_df["Multiple Lines"].unique(), default=churn_df["Multiple Lines"].unique())
selected_internet_service = st.sidebar.multiselect("Internet Service", churn_df["Internet Service"].unique(), default=churn_df["Internet Service"].unique())
selected_online_security = st.sidebar.multiselect("Online Security", churn_df["Online Security"].unique(), default=churn_df["Online Security"].unique())
selected_online_backup = st.sidebar.multiselect("Online Backup", churn_df["Online Backup"].unique(), default=churn_df["Online Backup"].unique())
selected_device_protection = st.sidebar.multiselect("Device Protection", churn_df["Device Protection"].unique(), default=churn_df["Device Protection"].unique())
selected_tech_support = st.sidebar.multiselect("Tech Support", churn_df["Tech Support"].unique(), default=churn_df["Tech Support"].unique())
selected_streaming_tv = st.sidebar.multiselect("Streaming TV", churn_df["Streaming TV"].unique(), default=churn_df["Streaming TV"].unique())
selected_streaming_movies = st.sidebar.multiselect("Streaming Movies", churn_df["Streaming Movies"].unique(), default=churn_df["Streaming Movies"].unique())
selected_contract = st.sidebar.multiselect("Contract", churn_df["Contract"].unique(), default=churn_df["Contract"].unique())
selected_paperless_billing = st.sidebar.multiselect("Paperless Billing", churn_df["Paperless Billing"].unique(), default=churn_df["Paperless Billing"].unique())
selected_payment_method = st.sidebar.multiselect("Payment Method", churn_df["Payment Method"].unique(), default=churn_df["Payment Method"].unique())

# --- SECTION 2: Filter Data ---
# Creating the filtered dataframe
filtered_df = churn_df[
    (churn_df["Gender"].isin(selected_gender)) &
    (churn_df["Senior Citizen"].isin(selected_senior_citizen)) &
    (churn_df["Partner"].isin(selected_partner)) &
    (churn_df["Dependents"].isin(selected_dependents)) &
    (churn_df["Phone Service"].isin(selected_phone_service)) &
    (churn_df["Multiple Lines"].isin(selected_multiple_lines)) &
    (churn_df["Internet Service"].isin(selected_internet_service)) &
    (churn_df["Online Security"].isin(selected_online_security)) &
    (churn_df["Online Backup"].isin(selected_online_backup)) &
    (churn_df["Device Protection"].isin(selected_device_protection)) &
    (churn_df["Tech Support"].isin(selected_tech_support)) &
    (churn_df["Streaming TV"].isin(selected_streaming_tv)) &
    (churn_df["Streaming Movies"].isin(selected_streaming_movies)) &
    (churn_df["Contract"].isin(selected_contract)) &
    (churn_df["Paperless Billing"].isin(selected_paperless_billing)) &
    (churn_df["Payment Method"].isin(selected_payment_method))
]

# Showing the filtered dataframe
st.subheader("Filtered Data")
st.write(filtered_df.head())

# --- SECTION 3: Churn Values Distribution ---
# Showing the distribution of churn values with the applied filters
st.subheader("Churn Distribution")
fig = plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_df, x="Churn Value")
plt.xticks([0, 1], ["Not Churned", "Churned"])
st.pyplot(fig)

# --- SECTION 4: Tenure Months Distribution --- 
# Showing the distribution of tenure months with the applied filters
st.subheader("Tenure Months Distribution")
fig = plt.figure(figsize=(10, 6))
sns.histplot(filtered_df["Tenure Months"], bins=churn_df["Tenure Months"].nunique(), kde=True)
st.pyplot(fig)

# --- SECTION 5: SHAP Feature Importances for Selected Customer ---
# Creating an input field for selecting a customer by index
st.subheader("SHAP Feature Importances for Selected Customer")
customer_index = st.number_input("Select a customer index:", min_value=0, max_value=len(X_test_df)-1, step=1)

# Showing the predicted churn probability and label
prediction_prob = best_model.predict_proba(X_test_df.iloc[[customer_index]])[0][1]
prediction_prob = prediction_prob.item()
prediction_label = best_model.predict(X_test_df.iloc[[customer_index]])[0]

st.markdown(f"**Predicted Churn Probability:** {prediction_prob:.2f}")
if prediction_label == 1:
    st.markdown(f"**Predicted Churn Label:** :red[Churn]")
else:
    st.markdown(f"**Predicted Churn Label:** :green[No Churn]")
    
# Creating a placeholder for the SHAP feature importances plot
shap_feature_importance_placeholder = st.empty()

# Caching function to get SHAP values for prediction to avoid unnecessary recomputing
@st.cache_data
def get_shap_values():
    explainer = shap.LinearExplainer(best_model, X_test_df)
    shap_values = explainer(X_test_df)
    return explainer, shap_values

explainer, shap_values = get_shap_values()

# Plotting the SHAP feature importances for the selected customer
fig = plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[customer_index])
shap_feature_importance_placeholder.pyplot(fig, clear_figure=True)