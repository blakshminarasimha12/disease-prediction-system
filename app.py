import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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
    y_text = df[target].astype(str).fillna("Unknown")
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_text), index=df.index)
    class_names = label_encoder.classes_

    X = df.drop(columns=[target]).copy()
    id_columns = [c for c in X.columns if c in ["patient_id", "patientid", "id", "serial_no", "serial_number"]]
    if id_columns:
        X = X.drop(columns=id_columns)

    for col in X.columns:
        if X[col].dtype == "object":
            numeric = pd.to_numeric(X[col], errors="coerce")
            if numeric.notna().mean() > 0.8:
                X[col] = numeric

    X = X.replace([np.inf, -np.inf], np.nan)
    return df, X, y, target, class_names

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, class_weight="balanced")
    }

@st.cache_resource
def train_models(data_key, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    trained = {}

    for name, estimator in get_models().items():
        preprocessor = build_preprocessor(X)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        if name == "Naive Bayes":
            model = Pipeline([
                ("preprocessor", build_preprocessor(X)),
                ("model", GaussianNB())
            ])

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, prediction, average="weighted", zero_division=0)
        })

        trained[name] = model

    return pd.DataFrame(results), trained, X_test, y_test

uploaded_file = st.sidebar.file_uploader("Upload Disease Dataset", type=["csv"])
model_name = st.sidebar.selectbox(
    "Prediction Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "SVM"]
)

try:
    raw_df = load_data(uploaded_file)
    df, X, y, target_column, class_names = prepare_data(raw_df)

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
best = results.loc[results["F1-Score"].idxmax()]

target_text = df[target_column].astype(str)
most_common = target_text.value_counts().index[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", total_patients)
c2.metric("Disease Classes", disease_count)
c3.metric("Most Common Disease", most_common)
c4.metric("Best Accuracy", f"{best['Accuracy'] * 100:.1f}%")
c5.metric("Best F1 Score", f"{best['F1-Score'] * 100:.1f}%")

st.divider()
st.header("📊 Healthcare Analytics Dashboard")

a, b = st.columns(2)

with a:
    st.subheader("1. Disease Distribution")
    counts = target_text.value_counts().head(15)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Disease")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with b:
    st.subheader("2. Disease Percentage")
    counts = target_text.value_counts().head(10)
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    st.pyplot(fig)
    plt.close(fig)

a, b = st.columns(2)

with a:
    st.subheader("3. Model Accuracy")
    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["Accuracy"] * 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)

with b:
    st.subheader("4. Model Performance")
    x_axis = np.arange(len(results))
    width = 0.22
    fig, ax = plt.subplots()
    ax.bar(x_axis - width * 1.5, results["Accuracy"] * 100, width, label="Accuracy")
    ax.bar(x_axis - width / 2, results["Precision"] * 100, width, label="Precision")
    ax.bar(x_axis + width / 2, results["Recall"] * 100, width, label="Recall")
    ax.bar(x_axis + width * 1.5, results["F1-Score"] * 100, width, label="F1")
    ax.set_xticks(x_axis, results["Model"], rotation=30)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

a, b = st.columns(2)

with a:
    st.subheader("5. Numerical Feature Distribution")
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        feature = st.selectbox("Select Numerical Feature", numeric_columns)
        fig, ax = plt.subplots()
        ax.hist(X[feature].dropna(), bins=25)
        ax.set_xlabel(feature)
        ax.set_ylabel("Records")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numerical features available.")

with b:
    st.subheader("6. Feature Statistics")
    if len(X.select_dtypes(include=np.number).columns) > 0:
        st.dataframe(X.describe().transpose().round(2), use_container_width=True)
    else:
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

a, b = st.columns(2)

with a:
    st.subheader("7. Feature Correlation")
    numeric = X.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(numeric.corr(), ax=ax, cmap="coolwarm", annot=False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numerical features are required.")

with b:
    st.subheader("8. Missing Values")
    missing = df.isnull().sum().sort_values(ascending=False).head(15)
    if missing.sum() > 0:
        fig, ax = plt.subplots()
        ax.barh(missing.index[::-1], missing.values[::-1])
        ax.set_xlabel("Missing Values")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.success("No missing values found.")

a, b = st.columns(2)

with a:
    st.subheader("9. Confusion Matrix")
    predictions = selected_model.predict(X_test)
    labels = sorted(np.unique(np.concatenate([y_test.values, predictions])))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    label_names = [class_names[int(i)] for i in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("Actual Disease")
    st.pyplot(fig)
    plt.close(fig)

with b:
    st.subheader("10. Top Disease Classes")
    top = target_text.value_counts().head(10)
    fig, ax = plt.subplots()
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Number of Records")
    st.pyplot(fig)
    plt.close(fig)

a, b = st.columns(2)

with a:
    st.subheader("11. Dataset Class Balance")
    balance = target_text.value_counts()
    fig, ax = plt.subplots()
    ax.plot(range(len(balance)), balance.values, marker="o")
    ax.set_xticks(range(len(balance)))
    ax.set_xticklabels(balance.index, rotation=45, ha="right")
    ax.set_ylabel("Records")
    st.pyplot(fig)
    plt.close(fig)

with b:
    st.subheader("12. Numeric Feature Boxplot")
    numeric = X.select_dtypes(include=np.number)
    if numeric.shape[1] > 0:
        box_features = st.multiselect(
            "Select Features",
            numeric.columns.tolist(),
            default=numeric.columns.tolist()[:min(5, len(numeric.columns))]
        )
        if box_features:
            box_data = [pd.to_numeric(numeric[c], errors="coerce").dropna().to_numpy() for c in box_features]
            valid_pairs = [(name, values) for name, values in zip(box_features, box_data) if len(values) > 0]
            if valid_pairs:
                valid_names = [name for name, _ in valid_pairs]
                valid_data = [values for _, values in valid_pairs]
                fig, ax = plt.subplots()
                ax.boxplot(valid_data)
                ax.set_xticks(range(1, len(valid_names) + 1))
                ax.set_xticklabels(valid_names, rotation=30, ha="right")
                ax.set_ylabel("Value")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Selected features contain no valid numeric values.")
    else:
        st.info("No numerical features available.")

a, b = st.columns(2)

with a:
    st.subheader("13. Numeric Feature Averages by Disease")
    numeric = X.select_dtypes(include=np.number)
    if numeric.shape[1] > 0:
        selected_avg = st.selectbox(
            "Feature for Disease Comparison",
            numeric.columns.tolist(),
            key="avg_feature"
        )
        temp = pd.DataFrame({
            "Disease": target_text.values,
            "Value": numeric[selected_avg].values
        })
        avg = temp.groupby("Disease")["Value"].mean().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots()
        ax.bar(avg.index, avg.values)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Average")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numerical features available.")

with b:
    st.subheader("14. Dataset Overview")
    overview = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Numerical Features",
            "Categorical Features",
            "Disease Classes",
            "Missing Cells"
        ],
        "Value": [
            len(df),
            len(df.columns),
            len(X.select_dtypes(include=np.number).columns),
            len(X.select_dtypes(exclude=np.number).columns),
            y.nunique(),
            int(df.isnull().sum().sum())
        ]
    })
    st.dataframe(overview, hide_index=True, use_container_width=True)

st.divider()
st.header("🔍 Disease Prediction")

st.write("Enter patient information below. The prediction is for educational demonstration only.")

input_values = {}
features = X.columns.tolist()
input_cols = st.columns(min(3, max(1, len(features))))

for i, feature in enumerate(features):
    col = input_cols[i % len(input_cols)]

    if pd.api.types.is_numeric_dtype(X[feature]):
        values = pd.to_numeric(X[feature], errors="coerce")
        default = float(values.median()) if values.notna().any() else 0.0
        input_values[feature] = col.number_input(
            feature.replace("_", " ").title(),
            value=default
        )
    else:
        values = X[feature].dropna().astype(str).unique().tolist()
        if values:
            input_values[feature] = col.selectbox(
                feature.replace("_", " ").title(),
                values
            )
        else:
            input_values[feature] = col.text_input(
                feature.replace("_", " ").title()
            )

if st.button("🏥 Predict Disease", type="primary"):
    input_df = pd.DataFrame([input_values])
    prediction = selected_model.predict(input_df)[0]
    prediction_name = class_names[int(prediction)]

    st.success(f"Predicted Disease: **{prediction_name}**")

    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Disease": [class_names[int(i)] for i in selected_model.classes_],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Prediction Probability")
        st.dataframe(
            probability_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True
        )

        top_probability = probability_df.head(10)
        fig, ax = plt.subplots()
        ax.barh(
            top_probability["Disease"][::-1],
            top_probability["Probability"][::-1] * 100
        )
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.header("📋 Model Evaluation")

evaluation = results.copy()

for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    evaluation[col] = (evaluation[col] * 100).round(2).astype(str) + "%"

st.dataframe(evaluation, use_container_width=True)

st.download_button(
    "⬇️ Download Model Results",
    results.to_csv(index=False).encode("utf-8"),
    "disease_model_results.csv",
    "text/csv"
)

st.download_button(
    "⬇️ Download Dataset",
    df.to_csv(index=False).encode("utf-8"),
    "disease_dataset.csv",
    "text/csv"
)

st.caption("Disease Prediction System | Python | Pandas | Scikit-learn | Streamlit")
