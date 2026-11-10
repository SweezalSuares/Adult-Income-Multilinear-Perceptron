
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Income Prediction MLP", page_icon="📊", layout="wide")

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "adult_train.csv")
TARGET = "Target"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None, path: str) -> pd.DataFrame:
    """Load adult_train.csv from an upload or from data/ and clean whitespace."""
    if file_bytes is not None:
        import io

        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    df[TARGET] = df[TARGET].str.replace(".", "", regex=False)
    return df


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, list(numeric_cols)), ("cat", categorical, list(categorical_cols))]
    )


def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)
def train_model(X_train, y_train, activation, units, epochs, batch_size, early_stop, seed):
    # Imported lazily so the app starts quickly.
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for u in units:
        model.add(Dense(u, activation=activation))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    callbacks = (
        [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        if early_stop
        else []
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )
    return model, history.history


def param_table(model, input_dim) -> pd.DataFrame:
    rows, prev = [], input_dim
    for i, layer in enumerate(model.layers, 1):
        n = layer.units
        rows.append(
            {
                "Layer": f"{i}: {layer.name}",
                "Neurons": n,
                "Activation": layer.activation.__name__,
                "Calculation": f"({prev} inputs x {n} neurons) + {n} biases",
                "Parameters": prev * n + n,
            }
        )
        prev = n
    return pd.DataFrame(rows)


def evaluate(model, X_test, y_test, threshold):
    proba = model.predict(X_test, verbose=0).flatten()
    pred = (proba >= threshold).astype(int)
    mse = mean_squared_error(y_test, proba)
    return {
        "proba": proba,
        "pred": pred,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "mae": mean_absolute_error(y_test, proba),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, proba),
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
upload = st.sidebar.file_uploader("Upload adult_train.csv", type="csv")

st.sidebar.subheader("Model")
activations = st.sidebar.multiselect(
    "Activation(s) to train and compare", ["relu", "tanh", "sigmoid", "elu"], default=["relu", "tanh"]
)
h1 = st.sidebar.slider("Hidden layer 1 neurons", 8, 256, 64, step=8)
h2 = st.sidebar.slider("Hidden layer 2 neurons (0 = none)", 0, 128, 32, step=8)
epochs = st.sidebar.slider("Epochs", 1, 100, 20)
batch_size = st.sidebar.select_slider("Batch size", [16, 32, 64, 128, 256], value=32)
early_stop = st.sidebar.checkbox("Early stopping (patience 3)", value=False)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
train_clicked = st.sidebar.button("🚀 Train model(s)", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
st.title("📊 Income Prediction with a Multi-Layer Perceptron")
st.caption("Adult / Census Income dataset · predicts whether income is >50K · Lab 6 notebook as a Streamlit app")

if upload is None and not os.path.exists(DEFAULT_CSV):
    st.info(
        "Upload **adult_train.csv** in the sidebar (or place it at `data/adult_train.csv`) to begin. "
        "It is the same file the notebook reads (32,561 rows, 15 columns)."
    )
    st.stop()

df = load_data(upload.getvalue() if upload else None, DEFAULT_CSV)
if TARGET not in df.columns:
    st.error(f"The CSV must contain a '{TARGET}' column. Found: {list(df.columns)}")
    st.stop()

X_raw = df.drop(columns=TARGET)
y = df[TARGET].map({"<=50K": 0, ">50K": 1})
if y.isna().any():
    st.error("Target column must contain only '<=50K' and '>50K' values.")
    st.stop()

numeric_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_raw.select_dtypes(include=["object"]).columns

# --------------------------------------------------------------------------- #
# Training (stored in session_state)
# --------------------------------------------------------------------------- #
if train_clicked:
    if not activations:
        st.sidebar.error("Pick at least one activation.")
    else:
        units = [h1] + ([h2] if h2 > 0 else [])
        # The split happens BEFORE fitting the preprocessor (no test-set leakage).
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_raw, y, test_size=test_size, random_state=int(seed), stratify=y
        )
        pre = build_preprocessor(numeric_cols, categorical_cols)
        X_tr = to_dense(pre.fit_transform(X_tr_raw))
        X_te = to_dense(pre.transform(X_te_raw))
        results = {}
        bar = st.progress(0.0, text="Training...")
        for i, act in enumerate(activations):
            bar.progress(i / len(activations), text=f"Training with {act}...")
            model, hist = train_model(
                X_tr, y_tr, act, units, epochs, batch_size, early_stop, int(seed)
            )
            results[act] = {"model": model, "history": hist}
        bar.empty()
        st.session_state["run"] = {
            "pre": pre,
            "X_te": X_te,
            "y_te": y_te,
            "n_features": X_tr.shape[1],
            "n_train": X_tr.shape[0],
            "results": results,
        }
        st.toast("Training complete", icon="✅")

run = st.session_state.get("run")

tab_data, tab_train, tab_eval, tab_predict, tab_about = st.tabs(
    ["📁 Data", "📈 Training", "🎯 Evaluation", "🔮 Predict", "ℹ️ About"]
)

# --------------------------------------------------------------------------- #
# Data tab
# --------------------------------------------------------------------------- #
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Features", X_raw.shape[1])
    c3.metric("Income >50K", f"{y.mean():.1%}")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Target balance")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df[TARGET].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d", hue=counts.index, legend=False)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Missing values")
        miss = df.isna().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.write("No missing values.")
        else:
            st.dataframe(miss.rename("Missing").to_frame(), use_container_width=True)
            st.caption("Imputed with median (numeric) / most frequent (categorical).")

    st.subheader("Feature explorer")
    col = st.selectbox("Column", list(X_raw.columns))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if col in numeric_cols:
        sns.histplot(data=df, x=col, hue=TARGET, bins=30, ax=ax, element="step")
    else:
        order = df[col].value_counts().index
        sns.countplot(data=df, y=col, hue=TARGET, order=order, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Preprocessing")
    st.markdown(
        f"- **Numeric ({len(numeric_cols)}):** {', '.join(numeric_cols)} → median imputation + `StandardScaler`\n"
        f"- **Categorical ({len(categorical_cols)}):** {', '.join(categorical_cols)} → most-frequent imputation + one-hot encoding"
    )

# --------------------------------------------------------------------------- #
# Training tab
# --------------------------------------------------------------------------- #
with tab_train:
    if not run:
        st.info("Choose settings in the sidebar and click **Train model(s)**.")
    else:
        st.write(
            f"Trained on **{run['n_train']:,}** rows (80/20 internal validation split), "
            f"**{run['n_features']}** input features after encoding."
        )
        for act, r in run["results"].items():
            h = r["history"]
            best_ep = int(np.argmin(h["val_loss"])) + 1
            st.subheader(f"{act.upper()} model")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final train loss", f"{h['loss'][-1]:.4f}")
            m2.metric("Final val loss", f"{h['val_loss'][-1]:.4f}")
            m3.metric("Best val loss", f"{min(h['val_loss']):.4f}", f"epoch {best_ep}", delta_color="off")
            gap = h["val_loss"][-1] - h["loss"][-1]
            m4.metric("Val − train loss", f"{gap:.4f}")
            if gap > 0.05 and best_ep < len(h["loss"]):
                st.warning(
                    f"Validation loss bottomed out at epoch {best_ep} and rose afterwards while training loss kept "
                    "falling, which is a sign of overfitting. Try fewer epochs, early stopping or a smaller network."
                )

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].plot(h["accuracy"], label="Train", marker="o", ms=3)
            axs[0].plot(h["val_accuracy"], label="Validation", marker="x", ms=4)
            axs[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
            axs[1].plot(h["loss"], label="Train", marker="o", ms=3)
            axs[1].plot(h["val_loss"], label="Validation", marker="x", ms=4)
            axs[1].axvline(best_ep - 1, ls="--", c="grey", lw=1, label="Best val loss")
            axs[1].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
            for a in axs:
                a.legend()
                a.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Architecture and parameter count**")
            pt = param_table(r["model"], run["n_features"])
            st.dataframe(pt, use_container_width=True, hide_index=True)
            st.caption(f"Total trainable parameters: **{int(pt['Parameters'].sum()):,}**")

# --------------------------------------------------------------------------- #
# Evaluation tab
# --------------------------------------------------------------------------- #
with tab_eval:
    if not run:
        st.info("Train a model first.")
    else:
        threshold = st.slider("Decision threshold for >50K", 0.1, 0.9, 0.5, 0.05)
        y_te = run["y_te"]
        evals = {a: evaluate(r["model"], run["X_te"], y_te, threshold) for a, r in run["results"].items()}

        st.subheader("Model comparison (test set)")
        comp = pd.DataFrame(
            {
                a: {k: e[k] for k in ["accuracy", "precision", "recall", "f1"]}
                for a, e in evals.items()
            }
        ).T
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)

        pick = st.selectbox("Inspect model", list(evals.keys()))
        e = evals[pick]
        a, b = st.columns(2)
        with a:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_te, e["pred"])
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"],
            )
            ax.set(xlabel="Predicted", ylabel="Actual")
            st.pyplot(fig)
            plt.close(fig)
        with b:
            st.markdown("**Classification report**")
            rep = classification_report(
                y_te, e["pred"], target_names=["Income <=50K", "Income >50K"], output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

        with st.expander("Regression-style metrics on predicted probabilities (as in the notebook)"):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("MAE", f"{e['mae']:.4f}")
            r2.metric("MSE", f"{e['mse']:.4f}")
            r3.metric("RMSE", f"{e['rmse']:.4f}")
            r4.metric("R²", f"{e['r2']:.4f}")
            st.caption(
                "These treat probabilities as continuous outputs. They are a calibration-style diagnostic; "
                "accuracy, precision, recall and F1 are the primary classification metrics."
            )

        with st.expander("Probability distribution"):
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.histplot(x=e["proba"], hue=y_te.values, bins=30, ax=ax, element="step")
            ax.axvline(threshold, c="k", ls="--")
            ax.set(xlabel="Predicted P(income >50K)")
            st.pyplot(fig)
            plt.close(fig)

# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
with tab_predict:
    if not run:
        st.info("Train a model first.")
    else:
        which = st.selectbox("Model", list(run["results"].keys()), key="pred_model")
        thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="pred_thr")
        with st.form("predict_form"):
            cols = st.columns(3)
            values = {}
            for i, c in enumerate(X_raw.columns):
                with cols[i % 3]:
                    if c in numeric_cols:
                        lo, hi = int(X_raw[c].min()), int(X_raw[c].max())
                        values[c] = st.number_input(c, lo, hi, int(X_raw[c].median()))
                    else:
                        opts = sorted(X_raw[c].dropna().unique())
                        mode = X_raw[c].mode().iloc[0]
                        values[c] = st.selectbox(c, opts, index=opts.index(mode))
            go = st.form_submit_button("Predict income class", type="primary")
        if go:
            row = pd.DataFrame([values])[list(X_raw.columns)]
            p = float(run["results"][which]["model"].predict(to_dense(run["pre"].transform(row)), verbose=0)[0][0])
            label = ">50K" if p >= thr else "<=50K"
            c1, c2 = st.columns(2)
            c1.metric("Predicted income", label)
            c2.metric("P(income >50K)", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
**Pipeline (mirrors the notebook)**

1. Load `adult_train.csv`, map target `<=50K → 0`, `>50K → 1`.
2. Preprocess: median impute + standardise numerics; most-frequent impute + one-hot encode categoricals (105 features on the full data).
3. Stratified train/test split.
4. MLP: `Dense(64) → Dense(32) → Dense(1, sigmoid)`, Adam, binary cross-entropy, 20 epochs, batch 32, 20% validation split.
5. Evaluate with accuracy, precision, recall, F1, confusion matrix and probability-level error metrics; compare ReLU vs Tanh.

**Difference from the notebook:** here the preprocessor is fitted on the training split only. The notebook fits it on the full
dataset before splitting, which leaks test-set statistics (scaling and category levels) into training. Expect results to differ slightly
from the notebook's 85.3% (ReLU) and 86.1% (Tanh).

Notebook: `Sweezal_Suares_547_lab6.ipynb`
        """
    )
