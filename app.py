import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="wide")

st.title("🏥 Disease Prediction System")
st.caption("Healthcare Analytics using Machine Learning")
st.warning("Educational decision-support application. This system is not a medical diagnosis and should not replace a qualified healthcare professional.")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    for path in ["data/disease_prediction.csv", "data/disease.csv", "disease_prediction.csv", "disease.csv"]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            pass
    raise FileNotFoundError("Dataset not found. Upload a CSV or place disease_prediction.csv inside data/.")

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def find_target(df):
    for col in ["disease", "target", "diagnosis", "label", "prognosis", "outcome"]:
        if col in df.columns:
            return col
    return df.columns[-1]

def prepare_data(raw):
    df = clean_columns(raw)
    target = find_target(df)
    df = df.dropna(axis=1, how="all").drop_duplicates()
    y = df[target].astype(str)
    X = df.drop(columns=[target]).copy()
    id_columns = [c for c in X.columns if c in ["patient_id", "patientid", "id", "serial_no", "serial_number"]]
    if id_columns:
        X = X.drop(columns=id_columns)
    for col in X.columns:
        if X[col].dtype == "object":
            numeric = pd.to_numeric(X[col], errors="coerce")
            if numeric.notna().mean() > 0.8:
                X[col] = numeric
            else:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].fillna("Unknown").astype(str))
    X = X.replace([np.inf, -np.inf], np.nan)
    return df, X, y, target

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"))
        ]),
        "Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced"))
        ])
    }

@st.cache_resource
def train_models(data_key, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    trained = {}
    rows = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, pred, average="weighted", zero_division=0)
        })
        trained[name] = model
    return pd.DataFrame(rows), trained, X_test, y_test

uploaded_file = st.sidebar.file_uploader("Upload Disease Dataset", type=["csv"])
model_name = st.sidebar.selectbox("Prediction Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "SVM"])

try:
    raw_df = load_data(uploaded_file)
    df, X, y, target_column = prepare_data(raw_df)
    if y.nunique() < 2:
        st.error("The dataset must contain at least two disease classes.")
        st.stop()
    data_key = str(df.shape) + str(df.head(20).to_dict())
    results, trained_models, X_test, y_test = train_models(data_key, X, y)
except Exception as e:
    st.error(f"Could not load/train the dataset: {e}")
    st.stop()

selected_model = trained_models[model_name]
total_patients = len(df)
disease_count = y.nunique()
most_common = y.value_counts().index[0]
best = results.loc[results["F1-Score"].idxmax()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Patients", total_patients)
c2.metric("Disease Classes", disease_count)
c3.metric("Most Common", most_common)
c4.metric("Best Accuracy", f"{best['Accuracy'] * 100:.1f}%")
c5.metric("Best F1 Score", f"{best['F1-Score'] * 100:.1f}%")

st.divider()
st.header("📊 Healthcare Analytics Dashboard")

a, b = st.columns(2)

with a:
    st.subheader("1. Disease Distribution")
    counts = y.value_counts().head(15)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Disease")
    ax.set_ylabel("Patients")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with b:
    st.subheader("2. Disease Percentage")
    counts = y.value_counts().head(10)
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

a, b = st.columns(2)

with a:
    st.subheader("3. Model Accuracy")
    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["Accuracy"] * 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

with b:
    st.subheader("4. Model Performance")
    x = np.arange(len(results))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(x - width, results["Precision"] * 100, width, label="Precision")
    ax.bar(x, results["Recall"] * 100, width, label="Recall")
    ax.bar(x + width, results["F1-Score"] * 100, width, label="F1 Score")
    ax.set_xticks(x, results["Model"], rotation=30)
    ax.set_ylabel("Score (%)")
    ax.legend()
    st.pyplot(fig)

a, b = st.columns(2)

with a:
    st.subheader("5. Numerical Feature Distribution")
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        feature = st.selectbox("Select Feature", numeric_columns)
        fig, ax = plt.subplots()
        ax.hist(X[feature].dropna(), bins=25)
        ax.set_xlabel(feature)
        ax.set_ylabel("Patients")
        st.pyplot(fig)

with b:
    st.subheader("6. Feature Statistics")
    st.dataframe(X.describe().transpose().round(2), use_container_width=True)

a, b = st.columns(2)

with a:
    st.subheader("7. Feature Correlation")
    numeric = X.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(numeric.corr(), ax=ax, annot=False, cmap="coolwarm")
        st.pyplot(fig)
    else:
        st.info("At least two numerical features are required.")

with b:
    st.subheader("8. Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    if len(missing):
        fig, ax = plt.subplots()
        ax.barh(missing.index[::-1], missing.values[::-1])
        ax.set_xlabel("Missing Values")
        st.pyplot(fig)
    else:
        st.success("No missing values found.")

a, b = st.columns(2)

with a:
    st.subheader("9. Confusion Matrix")
    predictions = selected_model.predict(X_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("Actual Disease")
    st.pyplot(fig)

with b:
    st.subheader("10. Top Disease Classes")
    top = y.value_counts().head(10)
    fig, ax = plt.subplots()
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Number of Patients")
    st.pyplot(fig)

st.divider()
st.header("🔍 Disease Prediction")
st.write("Enter patient information below. The prediction is for educational demonstration only.")

input_values = {}
features = X.columns.tolist()
cols = st.columns(min(3, max(1, len(features))))

for i, feature in enumerate(features):
    col = cols[i % len(cols)]
    if pd.api.types.is_numeric_dtype(X[feature]):
        value = float(X[feature].median()) if X[feature].notna().any() else 0.0
        input_values[feature] = col.number_input(feature.replace("_", " ").title(), value=value)
    else:
        values = X[feature].dropna().unique().tolist()
        input_values[feature] = col.selectbox(feature.replace("_", " ").title(), values)

if st.button("🏥 Predict Disease", type="primary"):
    input_df = pd.DataFrame([input_values])
    prediction = selected_model.predict(input_df)[0]
    st.success(f"Predicted Disease: **{prediction}**")
    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Disease": selected_model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.subheader("Prediction Probability")
        st.dataframe(probability_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
        top_probability = probability_df.head(10)
        fig, ax = plt.subplots()
        ax.barh(top_probability["Disease"][::-1], top_probability["Probability"][::-1] * 100)
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)

st.divider()
st.header("📋 Model Evaluation")

evaluation = results.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    evaluation[col] = (evaluation[col] * 100).round(2).astype(str) + "%"

st.dataframe(evaluation, use_container_width=True)

st.download_button("⬇️ Download Model Results", results.to_csv(index=False).encode("utf-8"), "disease_model_results.csv", "text/csv")
st.download_button("⬇️ Download Dataset", df.to_csv(index=False).encode("utf-8"), "disease_dataset.csv", "text/csv")

st.caption("Disease Prediction System | Python | Pandas | Scikit-learn | Streamlit")
