
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import xgboost as xgb

from pathlib import Path

# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Customer Churn & Retention",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_PATH = BASE_DIR / "Churn_Modelling_dataset.csv"

st.title("🏦 Smart Customer Churn Prediction & Retention System")
st.caption(
    "Machine Learning • ANN • Feature Engineering • Customer Segmentation"
)

# =========================================================
# 2. LOAD SAVED MODELS
# =========================================================

@st.cache_resource
def load_models():
    required_files = {
        "xgb": MODEL_DIR / "xgboost_churn_model.json",
        "preprocessor": MODEL_DIR / "churn_preprocessor.pkl",
        "cluster_preprocessor": MODEL_DIR / "cluster_preprocessor.pkl",
        "kmeans": MODEL_DIR / "kmeans_model.pkl",
    }

    missing = [
        str(path) for path in required_files.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(required_files["xgb"]))

    return {
        "xgb": xgb_model,
        "preprocessor": joblib.load(required_files["preprocessor"]),
        "cluster_preprocessor": joblib.load(
            required_files["cluster_preprocessor"]
        ),
        "kmeans": joblib.load(required_files["kmeans"]),
    }


try:
    models = load_models()
except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.info(
        "Check that you extracted saved_models.zip into a "
        "saved_models folder beside app.py."
    )
    st.stop()

xgb_model = models["xgb"]
preprocessor = models["preprocessor"]
cluster_preprocessor = models["cluster_preprocessor"]
kmeans_model = models["kmeans"]

# =========================================================
# 3. LOAD DATASET FOR HISTORICAL ANALYSIS
# =========================================================

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

df = None

if DATA_PATH.exists():
    df = load_dataset(DATA_PATH)

# =========================================================
# 4. FEATURE ENGINEERING
# Must match training-time feature definitions
# =========================================================

def engineer_features(customer):
    data = customer.copy()

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 30, 40, 50, 60, float("inf")],
        labels=[
            "Young", "Adult", "Middle_Aged",
            "Senior", "Older"
        ]
    )

    data["BalanceCategory"] = pd.cut(
        data["Balance"],
        bins=[
            -0.01, 0, 50000, 100000, 150000,
            float("inf")
        ],
        labels=[
            "Zero_Balance", "Low", "Medium",
            "High", "Very_High"
        ]
    )

    data["ProductsPerTenure"] = (
        data["NumOfProducts"] / (data["Tenure"] + 1)
    )

    return data


# Original features used by K-Means
CLUSTER_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

def predict_customer(customer):
    engineered = engineer_features(customer)

    # Use the same feature names/order as training
    X = engineered[preprocessor.feature_names_in_]
    X_processed = preprocessor.transform(X)

    probability = float(
        xgb_model.predict_proba(X_processed)[0, 1]
    )

    # Initial project thresholds; tune with validation data later
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    cluster_input = customer[CLUSTER_FEATURES]
    cluster_processed = cluster_preprocessor.transform(
        cluster_input
    )

    if hasattr(cluster_processed, "toarray"):
        cluster_processed = cluster_processed.toarray()

    cluster = int(
        kmeans_model.predict(cluster_processed)[0]
    )

    return probability, risk, cluster


def retention_recommendation(customer, risk):
    if risk == "Low Risk":
        return "Maintain engagement and monitor"

    actions = []

    if customer["IsActiveMember"].iloc[0] == 0:
        actions.append("Re-engagement campaign")

    if customer["NumOfProducts"].iloc[0] >= 3:
        actions.append("Review product suitability")

    if customer["Age"].iloc[0] >= 50:
        actions.append("Personalized customer support")

    if customer["Geography"].iloc[0] == "Germany":
        actions.append("Review regional service experience")

    if customer["Balance"].iloc[0] == 0:
        actions.append("Offer relevant banking products")

    if not actions:
        actions.append("Personalized retention outreach")

    return "; ".join(actions)


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard Overview",
        "Predict Customer Churn",
        "Customer Segmentation",
        "Churn Factors",
        "Model Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Project: Smart Customer Churn Prediction & Retention System"
)

# =========================================================
# 6. DASHBOARD OVERVIEW
# =========================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    if df is None:
        st.warning(
            "Place Churn_Modelling.csv beside app.py to enable "
            "historical dataset analytics."
        )
    else:
        total = len(df)
        churned = int(df["Exited"].sum())
        retained = total - churned
        churn_rate = 100 * churned / total

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Customers", f"{total:,}")
        c2.metric("Churned Customers", f"{churned:,}")
        c3.metric("Retained Customers", f"{retained:,}")
        c4.metric("Historical Churn Rate", f"{churn_rate:.2f}%")

        left, right = st.columns(2)

        with left:
            churn_counts = (
                df["Exited"]
                .value_counts()
                .reindex([0, 1], fill_value=0)
                .rename(index={0: "Retained", 1: "Churned"})
                .rename_axis("Customer Status")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                churn_counts,
                names="Customer Status",
                values="Customers",
                title="Historical Customer Churn",
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            churn_geo = (
                df.groupby("Geography")["Exited"]
                .mean()
                .mul(100)
                .reset_index(name="Churn Rate (%)")
            )

            fig = px.bar(
                churn_geo,
                x="Geography",
                y="Churn Rate (%)",
                title="Churn Rate by Geography",
                text_auto=".1f"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key observations")

        st.write(
            f"- Historical churn rate: **{churn_rate:.2f}%**."
        )
        st.write(
            "- Previous analysis identified age, number of products, "
            "membership activity and geography as useful predictive features."
        )
        st.write(
            "- Use the prediction page to evaluate an individual customer's "
            "estimated churn probability."
        )

# =========================================================
# 7. CUSTOMER CHURN PREDICTION
# =========================================================

elif page == "Predict Customer Churn":

    st.header("🔍 Predict Customer Churn")

    st.write(
        "Enter customer information to estimate churn probability "
        "and generate a retention recommendation."
    )

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input(
                "Credit Score", 300, 900, 650
            )
            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )
            age = st.number_input(
                "Age", 18, 100, 40
            )

        with col2:
            tenure = st.number_input(
                "Tenure (years)", 0, 10, 5
            )
            balance = st.number_input(
                "Balance", 0.0, 300000.0, 50000.0, step=1000.0
            )
            products = st.selectbox(
                "Number of Products", [1, 2, 3, 4]
            )

        with col3:
            has_card = st.selectbox(
                "Has Credit Card?", ["Yes", "No"]
            )
            active_member = st.selectbox(
                "Is Active Member?", ["Yes", "No"]
            )
            salary = st.number_input(
                "Estimated Salary",
                0.0, 250000.0, 100000.0, step=1000.0
            )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(active_member == "Yes"),
            "EstimatedSalary": salary
        }])

        try:
            probability, risk, cluster = predict_customer(customer)
            action = retention_recommendation(customer, risk)

            st.divider()
            st.subheader("Prediction Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}"
            )

            c2.metric("Risk Category", risk)

            c3.metric("Customer Segment", f"Cluster {cluster}")

            st.progress(probability)

            if risk == "High Risk":
                st.error(
                    "High estimated churn risk. Consider reviewing "
                    "this customer's needs and available retention options."
                )
            elif risk == "Medium Risk":
                st.warning(
                    "Moderate estimated churn risk. Consider proactive "
                    "engagement and monitoring."
                )
            else:
                st.success(
                    "Low estimated churn risk under the current threshold rules."
                )

            st.subheader("Suggested Retention Action")
            st.info(action)

            st.caption(
                "This probability is a model estimate, not a guarantee "
                "that the customer will leave. Risk categories use provisional "
                "thresholds and should be validated before business use."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# =========================================================
# 8. CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    if df is None:
        st.warning(
            "Place Churn_Modelling.csv beside app.py to display "
            "historical customer segment profiles."
        )
    else:
        if "Cluster" in df.columns:
            st.info(
                "The source CSV already contains a Cluster column. "
                "The dashboard will recalculate clusters from the saved model."
            )

        cluster_data = df[CLUSTER_FEATURES].copy()

        cluster_processed = cluster_preprocessor.transform(
            cluster_data
        )

        if hasattr(cluster_processed, "toarray"):
            cluster_processed = cluster_processed.toarray()

        segment_df = df.copy()
        segment_df["Cluster"] = kmeans_model.predict(
            cluster_processed
        )

        profile = segment_df.groupby("Cluster").agg(
            Customers=("Exited", "size"),
            Average_Age=("Age", "mean"),
            Average_Balance=("Balance", "mean"),
            Average_Products=("NumOfProducts", "mean"),
            Active_Member_Rate=("IsActiveMember", "mean"),
            Churn_Rate=("Exited", "mean")
        ).reset_index()

        profile["Churn Rate (%)"] = profile["Churn_Rate"] * 100
        profile["Active Member Rate (%)"] = (
            profile["Active_Member_Rate"] * 100
        )

        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                profile,
                x="Cluster",
                y="Churn Rate (%)",
                title="Historical Churn Rate by Cluster",
                text_auto=".1f"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                profile,
                x="Cluster",
                y="Customers",
                title="Customers per Segment",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment profiles")

        display_columns = [
            "Cluster", "Customers", "Average_Age",
            "Average_Balance", "Average_Products",
            "Active Member Rate (%)", "Churn Rate (%)"
        ]

        st.dataframe(
            profile[display_columns].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Cluster IDs are arbitrary labels. The observed churn rate "
            "describes the historical dataset, not an individual's future outcome."
        )

# =========================================================
# 9. IMPORTANT CHURN FACTORS
# =========================================================

elif page == "Churn Factors":

    st.header("📈 Important Churn Factors")

    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = xgb_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(24, len(importance_df)),
            value=min(15, len(importance_df))
        )

        fig = px.bar(
            importance_df.head(top_n).sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} XGBoost Feature Importances"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature importance table")

        st.dataframe(
            importance_df.head(top_n).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Feature importance measures how the fitted model uses features "
            "to make predictions. It does not prove that a feature causes churn. "
            "Age groups and raw age, for example, may represent overlapping information."
        )

    except Exception as exc:
        st.error(f"Could not calculate feature importance: {exc}")

# =========================================================
# 10. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("🤖 Model Performance Comparison")

    # Results from the completed experiment.
    # Update these values if you retrain or change model thresholds.
    comparison = pd.DataFrame([
        {
            "Model": "XGBoost",
            "Accuracy": 0.8660,
            "Precision": 0.7663,
            "Recall": 0.4914,
            "F1 Score": 0.5988,
            "ROC-AUC": 0.8659
        },
        {
            "Model": "Random Forest",
            "Accuracy": 0.8620,
            "Precision": 0.7811,
            "Recall": 0.4472,
            "F1 Score": 0.5688,
            "ROC-AUC": 0.8585
        },
        {
            "Model": "Decision Tree",
            "Accuracy": 0.8620,
            "Precision": 0.7549,
            "Recall": 0.4767,
            "F1 Score": 0.5843,
            "ROC-AUC": 0.8410
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.8310,
            "Precision": 0.7091,
            "Recall": 0.2875,
            "F1 Score": 0.4091,
            "ROC-AUC": 0.7968
        },
        {
            "Model": "ANN",
            "Accuracy": 0.7875,
            "Precision": 0.4862,
            "Recall": 0.7813,
            "F1 Score": 0.5994,
            "ROC-AUC": 0.8658
        }
    ])

    st.dataframe(
        comparison.sort_values(
            "ROC-AUC", ascending=False
        ).round(4),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Compare models using",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    fig = px.bar(
        comparison.sort_values(metric, ascending=False),
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison: {metric}"
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "XGBoost is the current baseline for the default classification "
        "threshold. ANN provides higher churn recall, but its precision is "
        "lower. The preferred model depends on retention costs and the desired "
        "trade-off between missed churners and false alarms."
    )

# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Customer Churn Prediction & Retention System | "
    "Academic project prototype | Predictions should be validated "
    "before use in real banking decisions."
)
