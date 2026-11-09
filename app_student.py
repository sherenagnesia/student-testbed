"""
Student Performance Data Mining Testbed (Streamlit)
EDA · Preprocessing · Modeling · Evaluation, semuanya dapat diatur dari sidebar.

Data: UCI Student Performance (Mathematics), file student-mat.csv.
  - Unggah lewat sidebar, ATAU letakkan student-mat.csv di folder yang sama dengan file ini.
  - Jika tidak ada keduanya, aplikasi memakai data SIMULASI (bukan data asli).

Jalankan:  streamlit run app_student.py
"""
import io
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Student Performance Data Mining Testbed", page_icon="🎓", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(HERE, "student-mat.csv")
GRADES = ["G1", "G2", "G3"]
SCEN_A = "A: Early warning (drop G1 & G2)"
SCEN_B = "B: Progressive tracking (keep G1 & G2)"
MODEL_NAMES = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVM (RBF)",
    "Naive Bayes (Gaussian)",
]

ATTR_DESC = {
    "school": "Student's school (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
    "sex": "Student's sex (F / M)",
    "age": "Student's age (15–22)",
    "address": "Home address type (U = urban, R = rural)",
    "famsize": "Family size (LE3 = ≤ 3, GT3 = > 3)",
    "Pstatus": "Parents' cohabitation status (T = together, A = apart)",
    "Medu": "Mother's education (0 = none … 4 = higher education)",
    "Fedu": "Father's education (0 = none … 4 = higher education)",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason to choose this school",
    "guardian": "Student's guardian",
    "traveltime": "Home-to-school travel time (1 = <15 min … 4 = >1 h)",
    "studytime": "Weekly study time (1 = <2 h … 4 = >10 h)",
    "failures": "Number of past class failures",
    "schoolsup": "Extra educational support from school",
    "famsup": "Family educational support",
    "paid": "Extra paid classes within the course subject",
    "activities": "Extra-curricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants to take higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Quality of family relationships (1–5)",
    "freetime": "Free time after school (1–5)",
    "goout": "Going out with friends (1–5)",
    "Dalc": "Workday alcohol consumption (1–5)",
    "Walc": "Weekend alcohol consumption (1–5)",
    "health": "Current health status (1–5)",
    "absences": "Number of school absences",
    "G1": "First-period grade (0–20)",
    "G2": "Second-period grade (0–20)",
    "G3": "Final grade (0–20); the target is derived from this",
}


# =============================================================================
# DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def make_simulated(n: int = 395, seed: int = 7) -> pd.DataFrame:
    """Data SIMULASI dengan 33 kolom seperti student-mat.csv (bukan data asli)."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)

    def pick(opts, p=None):
        return rng.choice(opts, n, p=p)

    def yn(p):
        return rng.choice(["yes", "no"], n, p=[p, 1 - p])

    studytime = np.clip(np.round(2 + 0.5 * ability + rng.normal(0, 0.8, n)), 1, 4).astype(int)
    failures = np.clip(rng.poisson(np.exp(-1.3 - 0.6 * ability)), 0, 3)
    absences = np.clip(rng.poisson(np.exp(1.6 - 0.25 * ability)), 0, 75)

    g1 = np.clip(np.round(11.4 + 3.0 * ability + 0.5 * (studytime - 2) - 1.0 * failures + rng.normal(0, 1.5, n)), 3, 19)
    g2 = np.clip(np.round(g1 + 0.2 * ability + rng.normal(0, 1.2, n)), 0, 19)
    g3 = np.clip(np.round(g2 + rng.normal(0.2, 1.2, n)), 0, 20)
    dropout = (rng.random(n) < 0.08) & (g2 <= 10)  # sebagian siswa putus: G3 = 0
    g3 = np.where(dropout, 0, g3)

    return pd.DataFrame(
        {
            "school": pick(["GP", "MS"], [0.88, 0.12]),
            "sex": pick(["F", "M"], [0.53, 0.47]),
            "age": rng.choice([15, 16, 17, 18, 19, 20, 21, 22], n, p=[0.21, 0.26, 0.25, 0.21, 0.04, 0.01, 0.01, 0.01]),
            "address": pick(["U", "R"], [0.78, 0.22]),
            "famsize": pick(["GT3", "LE3"], [0.71, 0.29]),
            "Pstatus": pick(["T", "A"], [0.9, 0.1]),
            "Medu": rng.integers(0, 5, n),
            "Fedu": rng.integers(0, 5, n),
            "Mjob": pick(["at_home", "health", "other", "services", "teacher"], [0.15, 0.09, 0.36, 0.26, 0.14]),
            "Fjob": pick(["at_home", "health", "other", "services", "teacher"], [0.05, 0.05, 0.55, 0.28, 0.07]),
            "reason": pick(["course", "home", "other", "reputation"], [0.37, 0.28, 0.09, 0.26]),
            "guardian": pick(["mother", "father", "other"], [0.69, 0.23, 0.08]),
            "traveltime": rng.choice([1, 2, 3, 4], n, p=[0.65, 0.27, 0.06, 0.02]),
            "studytime": studytime,
            "failures": failures,
            "schoolsup": yn(0.13),
            "famsup": yn(0.61),
            "paid": yn(0.46),
            "activities": yn(0.51),
            "nursery": yn(0.79),
            "higher": np.where(rng.random(n) < 1 / (1 + np.exp(-(3.5 + 0.8 * ability))), "yes", "no"),
            "internet": yn(0.83),
            "romantic": yn(0.33),
            "famrel": rng.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.05, 0.17, 0.49, 0.27]),
            "freetime": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.16, 0.4, 0.29, 0.1]),
            "goout": rng.choice([1, 2, 3, 4, 5], n, p=[0.06, 0.26, 0.33, 0.22, 0.13]),
            "Dalc": rng.choice([1, 2, 3, 4, 5], n, p=[0.7, 0.19, 0.07, 0.02, 0.02]),
            "Walc": rng.choice([1, 2, 3, 4, 5], n, p=[0.38, 0.22, 0.2, 0.13, 0.07]),
            "health": rng.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.11, 0.23, 0.17, 0.37]),
            "absences": absences,
            "G1": g1.astype(int),
            "G2": g2.astype(int),
            "G3": g3.astype(int),
        }
    )


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """student-mat.csv memakai pemisah ';'. Coba ';', ',' lalu deteksi otomatis."""
    last = None
    for sep in (";", ",", None):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python" if sep is None else "c")
            if df.shape[1] > 1:
                df.columns = [str(c).strip().strip('"') for c in df.columns]
                return df
        except Exception as e:  # noqa: BLE001
            last = e
    raise ValueError(f"CSV tidak dapat dibaca: {last}")


def get_data(upload):
    """Prioritas: file unggahan > student-mat.csv lokal > data simulasi."""
    if upload is not None:
        return read_csv_bytes(upload.getvalue()), f"Uploaded file: {upload.name}", False
    if os.path.exists(DEFAULT_FILE):
        with open(DEFAULT_FILE, "rb") as f:
            return read_csv_bytes(f.read()), "Local file: student-mat.csv", False
    return make_simulated(), "SIMULATED data (not the real UCI file)", True


# =============================================================================
# PIPELINE
# =============================================================================
def build_xy(df: pd.DataFrame, threshold: int, scenario: str):
    y = (df["G3"] >= threshold).astype(int)
    drop = ["G3"] + (["G1", "G2"] if scenario.startswith("A") else [])
    return df.drop(columns=drop), y


def split_types(X: pd.DataFrame):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocessor(X: pd.DataFrame, scaler: str) -> ColumnTransformer:
    num, cat = split_types(X)
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scaler == "MinMaxScaler":
        num_steps.append(("sc", MinMaxScaler()))
    elif scaler == "StandardScaler":
        num_steps.append(("sc", StandardScaler()))
    parts = []
    if num:
        parts.append(("num", Pipeline(num_steps), num))
    if cat:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(parts)


def clean_names(names) -> list:
    return [str(n).split("__", 1)[-1] for n in names]


def build_model(name: str, p: dict, seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(C=p["lr_c"], max_iter=3000)
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=p["knn_k"])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["dt_depth"], random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["rf_n"], random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=p["gb_n"], learning_rate=p["gb_lr"], random_state=seed)
    if name == "SVM (RBF)":
        return SVC(C=p["svm_c"], kernel="rbf")  # skor ROC dari decision_function
    if name == "Naive Bayes (Gaussian)":
        return GaussianNB()
    raise ValueError(name)


@st.cache_data(show_spinner=False)
def prepare(df, threshold, scenario, scaler, test_size, seed):
    """Hasil preprocessing untuk ditampilkan di tab Preprocessing."""
    X, y = build_xy(df, threshold, scenario)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    ct = make_preprocessor(X_tr, scaler)
    A_tr = ct.fit_transform(X_tr)
    A_te = ct.transform(X_te)
    names = clean_names(ct.get_feature_names_out())
    num, cat = split_types(X_tr)
    return {
        "X_train": pd.DataFrame(A_tr, columns=names),
        "X_test": pd.DataFrame(A_te, columns=names),
        "y_train": y_tr.values,
        "y_test": y_te.values,
        "num": num,
        "binary": [c for c in cat if X[c].nunique(dropna=True) == 2],
        "nominal": [c for c in cat if X[c].nunique(dropna=True) != 2],
        "n_raw": X.shape[1],
        "dropped": ["G3"] + (["G1", "G2"] if scenario.startswith("A") else []),
    }


@st.cache_data(show_spinner="Training models…")
def run_pipeline(df, threshold, scenario, scaler, test_size, k, seed, models, params):
    X, y = build_xy(df, threshold, scenario)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    k_eff = int(min(k, np.bincount(y_tr).min()))
    skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed) if k_eff >= 2 else None

    rows, cv_f1, cms, rocs, reps, imps = [], {}, {}, {}, {}, {}
    for name in models:
        pipe = Pipeline([("prep", make_preprocessor(X_tr, scaler)), ("model", build_model(name, params, seed))])
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        fit_t = time.time() - t0
        pred = pipe.predict(X_te)
        score = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_te)

        cv_acc = cv_f = np.array([np.nan])
        if skf is not None:
            cv = cross_validate(pipe, X_tr, y_tr, cv=skf, scoring=["accuracy", "f1"])
            cv_acc, cv_f = cv["test_accuracy"], cv["test_f1"]
        cv_f1[name] = cv_f

        fpr, tpr, _ = roc_curve(y_te, score)
        rocs[name] = (fpr, tpr)
        cms[name] = confusion_matrix(y_te, pred, labels=[0, 1])
        reps[name] = pd.DataFrame(
            classification_report(y_te, pred, target_names=["Fail", "Pass"], output_dict=True, zero_division=0)
        ).T
        m = pipe.named_steps["model"]
        names = clean_names(pipe.named_steps["prep"].get_feature_names_out())
        if hasattr(m, "feature_importances_"):
            imps[name] = pd.Series(m.feature_importances_, index=names)
        elif hasattr(m, "coef_"):
            imps[name] = pd.Series(np.abs(np.asarray(m.coef_)).ravel(), index=names)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_te, pred),
                "Precision": precision_score(y_te, pred, zero_division=0),
                "Recall": recall_score(y_te, pred, zero_division=0),
                "F1": f1_score(y_te, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_te, score),
                "CV Acc (mean)": np.nanmean(cv_acc),
                "CV F1 (mean)": np.nanmean(cv_f),
                "CV F1 (std)": np.nanstd(cv_f),
                "Fit time (s)": fit_t,
            }
        )
    return {
        "metrics": pd.DataFrame(rows).set_index("Model"),
        "cv_f1": cv_f1,
        "cm": cms,
        "roc": rocs,
        "report": reps,
        "imp": imps,
        "k_eff": k_eff,
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }


# =============================================================================
# UI : SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("⚙️ Testbed Configuration")
    upload = st.file_uploader("Upload student-mat.csv (optional)", type=["csv"])
    threshold = st.slider("Pass threshold on G3 (0-20)", 0, 20, 10)
    st.divider()
    scenario = st.radio("Leakage scenario", [SCEN_A, SCEN_B])
    scaler = st.selectbox("Feature scaler", ["MinMaxScaler", "StandardScaler", "None"])
    test_size = st.slider("Test size (holdout)", 0.10, 0.40, 0.20, 0.05)
    k_folds = st.slider("Cross-validation folds (K)", 3, 10, 5)
    seed = int(st.number_input("Random seed", 0, 9999, 42))
    st.divider()
    models = st.multiselect("Models (choose at least 2)", MODEL_NAMES, default=["Logistic Regression", "Random Forest", "Gradient Boosting"])
    with st.expander("Hyperparameters"):
        params = {
            "lr_c": st.number_input("Logistic Regression: C", 0.01, 100.0, 1.0),
            "knn_k": st.slider("KNN: neighbours (k)", 1, 25, 5),
            "dt_depth": st.slider("Decision Tree: max depth", 1, 20, 5),
            "rf_n": st.slider("Random Forest: trees", 20, 500, 200, 10),
            "gb_n": st.slider("Gradient Boosting: estimators", 20, 400, 100, 10),
            "gb_lr": st.number_input("Gradient Boosting: learning rate", 0.01, 1.0, 0.1),
            "svm_c": st.number_input("SVM: C", 0.01, 100.0, 1.0),
        }

# =============================================================================
# UI : DATA & VALIDASI
# =============================================================================
try:
    df, source, simulated = get_data(upload)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()

missing_cols = [c for c in GRADES if c not in df.columns]
if missing_cols:
    st.error(f"This testbed needs the grade columns G1, G2 and G3. Missing: {missing_cols}")
    st.stop()
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
df = df.dropna(subset=["G3"]).reset_index(drop=True)

y_all = (df["G3"] >= threshold).astype(int)
if y_all.value_counts().min() < 10:
    st.error("This pass threshold leaves fewer than 10 students in one class. Move the threshold slider.")
    st.stop()

view = df.copy()
view["Result"] = np.where(y_all == 1, "Pass", "Fail")
view["Pass"] = y_all

st.title("🎓 Student Performance Data Mining Testbed")
st.caption(
    "End-to-end pipeline on the UCI Student Performance (Mathematics) dataset — "
    "EDA · Preprocessing · Modeling · Evaluation, all configurable from the sidebar."
)

t_over, t_eda, t_prep, t_model, t_eval = st.tabs(["📋 Overview", "🔍 EDA", "🛠️ Preprocessing", "⚙️ Modeling", "📊 Evaluation"])

# =============================================================================
# TAB 1 : OVERVIEW
# =============================================================================
with t_over:
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing values", int(df.isna().sum().sum()))
    c[3].metric("Pass rate", f"{y_all.mean():.1%}")
    st.caption(f"Data source: {source}")
    if simulated:
        st.warning(
            "No student-mat.csv found, so a SIMULATED dataset with the same 33 columns is used for demonstration. "
            "Upload the real student-mat.csv from the sidebar (UCI Student Performance dataset) for real results."
        )

    st.subheader("Raw data sample")
    st.dataframe(df.head(10))

    st.subheader("About the dataset")
    st.markdown(
        "The dataset describes student achievement in secondary education at two Portuguese schools. "
        "Attributes cover demographics, family background, school-related and social factors. "
        "**G1**, **G2** and **G3** are the first-period, second-period and final grades (0–20). "
        f"The target used here is **Pass = G3 ≥ {threshold}**, so this is a binary classification task."
    )

    st.subheader("Data dictionary")
    dd = pd.DataFrame(
        {
            "Attribute": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Role": [
                "Target basis" if c_ == "G3" else ("Feature (scenario B only)" if c_ in ("G1", "G2") else "Feature")
                for c_ in df.columns
            ],
            "Unique": df.nunique().values,
            "Example": [", ".join(map(str, df[c_].dropna().unique()[:3])) for c_ in df.columns],
            "Description": [ATTR_DESC.get(c_, "") for c_ in df.columns],
        }
    )
    st.dataframe(dd, hide_index=True)

    st.subheader("Target distribution")
    vc = view["Result"].value_counts().reset_index()
    vc.columns = ["Result", "Students"]
    st.plotly_chart(px.pie(vc, names="Result", values="Students", hole=0.4))

# =============================================================================
# TAB 2 : EDA
# =============================================================================
with t_eda:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c_ for c_ in df.columns if c_ not in num_cols]

    st.subheader("1. Target and grades")
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="G3", nbins=21, title="Final grade (G3) distribution")
    fig.add_vline(x=threshold - 0.5, line_dash="dash", annotation_text=f"pass threshold = {threshold}")
    c1.plotly_chart(fig)
    c2.plotly_chart(px.scatter(view, x="G2", y="G3", color="Result", opacity=0.7, title="G2 vs G3"))
    gc = df[GRADES].corr()
    st.markdown(
        f"Correlation G1–G3 = **{gc.loc['G1', 'G3']:.2f}**, G2–G3 = **{gc.loc['G2', 'G3']:.2f}**. "
        "Earlier grades almost give away the final grade, which is why the sidebar has two leakage scenarios: "
        "**A** predicts without G1/G2 (early warning), **B** keeps them (progressive tracking)."
    )
    zero_g3 = int((df["G3"] == 0).sum())
    if zero_g3:
        st.info(f"{zero_g3} students have G3 = 0. These are usually students who dropped out or skipped the final exam.")

    st.subheader("2. Data quality")
    q = st.columns(4)
    q[0].metric("Missing values", int(df.isna().sum().sum()))
    q[1].metric("Duplicate rows", int(df.duplicated().sum()))
    q[2].metric("Constant columns", int((df.nunique() <= 1).sum()))
    cnt = y_all.value_counts()
    q[3].metric("Class ratio (max/min)", f"{cnt.max() / cnt.min():.2f}")

    desc = df[num_cols].describe().T
    desc["skew"] = df[num_cols].skew()
    desc["outliers (IQR)"] = [
        int(((df[c_] < df[c_].quantile(0.25) - 1.5 * (df[c_].quantile(0.75) - df[c_].quantile(0.25)))
             | (df[c_] > df[c_].quantile(0.75) + 1.5 * (df[c_].quantile(0.75) - df[c_].quantile(0.25)))).sum())
        for c_ in num_cols
    ]
    st.markdown("**Numeric summary**")
    st.dataframe(desc.round(2))

    st.subheader("3. Numeric features vs result")
    nc = st.selectbox("Numeric feature", [c_ for c_ in num_cols if c_ != "G3"], index=num_cols.index("absences") if "absences" in num_cols else 0)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(view, x=nc, color="Result", barmode="overlay", opacity=0.7, title=f"{nc} by result"))
    b.plotly_chart(px.box(view, x="Result", y=nc, color="Result", title=f"{nc} by result"))
    if df[nc].nunique() <= 8:
        g = view.groupby(nc)["Pass"].agg(["mean", "count"]).reset_index()
        fig = px.bar(g, x=nc, y="mean", text="count", title=f"Pass rate by {nc} (labels = number of students)")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Pass rate")
        st.plotly_chart(fig)

    st.subheader("4. Categorical features vs result")
    if cat_cols:
        cc = st.selectbox("Categorical feature", cat_cols)
        g = view.groupby(cc)["Pass"].agg(["mean", "count"]).reset_index()
        fig = px.bar(g, x=cc, y="mean", text="count", title=f"Pass rate by {cc} (labels = number of students)")
        fig.add_hline(y=float(y_all.mean()), line_dash="dash", annotation_text="overall pass rate")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Pass rate")
        st.plotly_chart(fig)
    else:
        st.caption("No categorical columns.")

    st.subheader("5. Correlations")
    corr = df[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation matrix (Pearson)"))
    withg3 = corr["G3"].drop("G3").sort_values()
    fig = px.bar(withg3, orientation="h", title="Correlation of each numeric feature with G3")
    fig.update_layout(showlegend=False, xaxis_title="Correlation", yaxis_title="")
    st.plotly_chart(fig)

# =============================================================================
# TAB 3 : PREPROCESSING
# =============================================================================
with t_prep:
    st.header("Preprocessing")
    prep = prepare(df, threshold, scenario, scaler, test_size, seed)
    st.markdown(
        f"""
**Configuration (from the sidebar)**
- Target: `Pass = 1 if G3 ≥ {threshold}`, else `Fail = 0`
- Scenario: **{scenario}**; columns removed from the features: `{', '.join(prep['dropped'])}`
- Scaler on numeric features: **{scaler}**
- Split: stratified holdout, test size **{test_size:.2f}**, seed {seed}
- Cross-validation: **{k_folds}** folds, run on the training part only
"""
    )
    c = st.columns(4)
    c[0].metric("Raw features", prep["n_raw"])
    c[1].metric("Features after encoding", prep["X_train"].shape[1])
    c[2].metric("Train rows", len(prep["y_train"]))
    c[3].metric("Test rows", len(prep["y_test"]))

    st.subheader("Steps")
    st.markdown(
        f"1. **Target creation** from G3 and removal of G3 from the features (otherwise the target leaks).\n"
        f"2. **Scenario filter**: {'G1 and G2 are dropped' if scenario.startswith('A') else 'G1 and G2 are kept as features'}.\n"
        f"3. **Stratified split** so Pass/Fail proportions stay the same in train and test.\n"
        f"4. **Missing values**: median (numeric) and most frequent (categorical); the dataset itself has none if the count above is 0.\n"
        f"5. **Encoding**: {len(prep['binary'])} binary columns become one 0/1 column, {len(prep['nominal'])} nominal columns are one-hot encoded.\n"
        f"6. **Scaling**: {scaler} applied to {len(prep['num'])} numeric columns.\n\n"
        "All statistics (median, scaling range, categories) are learned from the training data only, then applied to the test data, "
        "so no information leaks from test to train."
    )

    g1, g2, g3 = st.columns(3)
    g1.markdown("**Numeric**")
    g1.write(", ".join(prep["num"]) or "-")
    g2.markdown("**Binary**")
    g2.write(", ".join(prep["binary"]) or "-")
    g3.markdown("**Nominal (one-hot)**")
    g3.write(", ".join(prep["nominal"]) or "-")

    st.subheader("Processed training data (first 15 rows)")
    st.dataframe(prep["X_train"].head(15))

    dist = pd.DataFrame(
        {
            "Set": ["Train"] * 2 + ["Test"] * 2,
            "Result": ["Fail", "Pass"] * 2,
            "Students": [
                int((prep["y_train"] == 0).sum()), int((prep["y_train"] == 1).sum()),
                int((prep["y_test"] == 0).sum()), int((prep["y_test"] == 1).sum()),
            ],
        }
    )
    st.plotly_chart(px.bar(dist, x="Set", y="Students", color="Result", barmode="group", title="Class balance in train vs test"))

# =============================================================================
# TAB 4 : MODELING
# =============================================================================
with t_model:
    st.header("Modeling")
    if len(models) < 2:
        st.warning("Choose at least 2 models in the sidebar.")
    else:
        res = run_pipeline(df, threshold, scenario, scaler, test_size, k_folds, seed, tuple(models), params)
        st.markdown(
            f"Models are trained on **{res['n_train']}** students and evaluated with **{res['k_eff']}-fold stratified cross-validation** "
            "(preprocessing is refit inside every fold)."
        )
        hp = {
            "Logistic Regression": f"C = {params['lr_c']}",
            "K-Nearest Neighbors": f"k = {params['knn_k']}",
            "Decision Tree": f"max_depth = {params['dt_depth']}",
            "Random Forest": f"n_estimators = {params['rf_n']}",
            "Gradient Boosting": f"n_estimators = {params['gb_n']}, learning_rate = {params['gb_lr']}",
            "SVM (RBF)": f"C = {params['svm_c']}",
            "Naive Bayes (Gaussian)": "default",
        }
        st.dataframe(pd.DataFrame({"Model": models, "Hyperparameters": [hp[m_] for m_ in models]}), hide_index=True)

        st.subheader("Cross-validation results (training data)")
        cvt = res["metrics"][["CV Acc (mean)", "CV F1 (mean)", "CV F1 (std)", "Fit time (s)"]].round(4)
        st.dataframe(cvt)
        long = pd.DataFrame([(m_, v) for m_, arr in res["cv_f1"].items() for v in arr], columns=["Model", "F1 per fold"]).dropna()
        if not long.empty:
            st.plotly_chart(px.box(long, x="Model", y="F1 per fold", points="all", title="F1 across cross-validation folds"))

# =============================================================================
# TAB 5 : EVALUATION
# =============================================================================
with t_eval:
    st.header("Evaluation")
    if len(models) < 2:
        st.warning("Choose at least 2 models in the sidebar.")
    else:
        res = run_pipeline(df, threshold, scenario, scaler, test_size, k_folds, seed, tuple(models), params)
        mt = res["metrics"]
        show_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

        st.subheader("Holdout test results (positive class = Pass)")
        st.dataframe(mt[show_cols].round(4).style.highlight_max(axis=0, color="#b7e4c7"))
        long = mt[show_cols].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1.05]))
        st.download_button("⬇ Download metrics (CSV)", mt.round(4).to_csv().encode("utf-8"), "evaluation_metrics.csv", "text/csv")

        best = mt["F1"].idxmax()
        st.success(f"🏆 Best model by F1: **{best}** (F1 {mt.loc[best, 'F1']:.3f}, accuracy {mt.loc[best, 'Accuracy']:.3f}, ROC-AUC {mt.loc[best, 'ROC-AUC']:.3f}).")

        st.subheader("ROC curves")
        fig = go.Figure()
        for m_, (fpr, tpr) in res["roc"].items():
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m_} (AUC {mt.loc[m_, 'ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig)

        st.subheader("Per-model details")
        pick = st.selectbox("Model", models, index=models.index(best))
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.imshow(res["cm"][pick], x=["Fail", "Pass"], y=["Fail", "Pass"], text_auto=True, color_continuous_scale="Blues",
                      labels=dict(x="Predicted", y="Actual", color="Students"), title=f"Confusion matrix: {pick}")
        )
        c2.markdown("**Classification report**")
        c2.dataframe(res["report"][pick].round(3))
        if pick in res["imp"]:
            fi = res["imp"][pick].sort_values(ascending=False).head(15)[::-1]
            fig = px.bar(fi, orientation="h", title=f"Top 15 most influential features: {pick}")
            fig.update_layout(showlegend=False, xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig)
        else:
            st.caption("This model type does not expose feature importance.")

        st.subheader("Scenario comparison: A (early warning) vs B (progressive tracking)")
        ra = run_pipeline(df, threshold, SCEN_A, scaler, test_size, k_folds, seed, tuple(models), params)["metrics"]["F1"]
        rb = run_pipeline(df, threshold, SCEN_B, scaler, test_size, k_folds, seed, tuple(models), params)["metrics"]["F1"]
        comp = pd.DataFrame(
            {"Model": list(ra.index) * 2, "Scenario": ["A: drop G1 & G2"] * len(ra) + ["B: keep G1 & G2"] * len(rb), "F1": list(ra.values) + list(rb.values)}
        )
        st.plotly_chart(px.bar(comp, x="Model", y="F1", color="Scenario", barmode="group", range_y=[0, 1.05]))
        gain = float((rb - ra).mean())
        st.markdown(
            f"- Average F1 with G1 & G2 (B) minus without them (A): **{gain:+.3f}**.\n"
            "- A large gain means the earlier grades carry most of the predictive power; the model is then mostly repeating the grade trend. "
            "Scenario A is the more honest test of whether background factors can warn about failing students early."
        )
