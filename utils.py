import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


# =======================================
# === FEATURE DEFINITIONS ===============
# =======================================
def get_feature_info():
    numerical_cols = ['Age', 'Gravida', 'ANC_followup']
    categorical_cols = [
        'Baby_position', 'Gest_Hypertension', 'Residence', 'Twin',
        'Chronic_diseases', 'Infections', 'APH', 'Uterine_problem',
        'Anemia', 'FHR_Abnormality', 'PTB_history', 'Still_birth_history',
        'Abortion_history', 'Preclampsia_Eclampsia', 'Prev_C_section', 'PROM'
    ]
    binary = ['No', 'Yes']

    categorical_options = {
        'Baby_position': ['Wrong', 'Right'],
        'Gest_Hypertension': binary,
        'Residence': ['Rural', 'Urban'],
        'Twin': binary,
        'Chronic_diseases': binary,
        'Infections': binary,
        'APH': binary,
        'Uterine_problem': binary,
        'Anemia': binary,
        'FHR_Abnormality': binary,
        'PTB_history': binary,
        'Still_birth_history': binary,
        'Abortion_history': binary,
        'Preclampsia_Eclampsia': binary,
        'Prev_C_section': binary,
        'PROM': binary
    }

    feature_order = [
        'Age', 'Baby_position', 'PROM', 'Residence', 'Twin', 'Gest_Hypertension',
        'Chronic_diseases', 'Infections', 'APH', 'Uterine_problem', 'Anemia',
        'FHR_Abnormality', 'PTB_history', 'Still_birth_history', 'Abortion_history',
        'Gravida', 'Preclampsia_Eclampsia', 'ANC_followup', 'Prev_C_section'
    ]

    return {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'categorical_options': categorical_options,
        'feature_order': feature_order,
        'class_names': ['Low Risk (Term)', 'High Risk (PTB)']
    }


# =======================================
# === ENCODING MAP ======================
# =======================================
def get_categorical_encoding_map(feature_info):
    enc_map = {}
    for feature, opts in feature_info['categorical_options'].items():
        enc_map[feature] = {opt: i for i, opt in enumerate(opts)}
    return enc_map


# =======================================
# === PREPROCESS INPUT ==================
# =======================================
def preprocess_input(raw_input_data, feature_info):
    df = pd.DataFrame([raw_input_data])
    df.columns = [c.replace(' ', '_').replace('/', '_') for c in df.columns]

    enc_map = get_categorical_encoding_map(feature_info)

    for col in feature_info['categorical_cols']:
        if col in df.columns:
            df[col] = df[col].map(enc_map[col])
        else:
            df[col] = 0

    for col in feature_info['numerical_cols']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            df[col] = 0.0

    for col in feature_info['feature_order']:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_info['feature_order']]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df


# =======================================
# === LOAD ALL ASSETS ===================
# =======================================
def load_all_assets(model_path, shap_path, lime_path, xtrain_path="X_train.csv"):
    print("🔄 Loading model and explanation assets...")

    # --- Load model ---
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")

    # --- Load SHAP background ---
    shap_bg = None
    try:
        shap_bg = joblib.load(shap_path)
        if isinstance(shap_bg, dict):
            for key in ["data", "background", "X_train", "df", "background_data"]:
                if key in shap_bg:
                    shap_bg = shap_bg[key]
                    break
            else:
                shap_bg = None
        elif hasattr(shap_bg, "data"):
            shap_bg = shap_bg.data
    except Exception as e:
        print(f"⚠️ Could not load SHAP background: {e}")
        shap_bg = None

    # --- Fallback: load from X_train.csv ---
    if shap_bg is None:
        try:
            shap_bg = pd.read_csv(xtrain_path)
            print(f"✅ Using {xtrain_path} as SHAP background.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load SHAP background: {e}")

    # --- Load or rebuild LIME explainer ---
    lime_obj = joblib.load(lime_path)
    if isinstance(lime_obj, dict):
        # Handle different storage formats
        if "explainer" in lime_obj:
            lime_obj = lime_obj["explainer"]
        elif "lime" in lime_obj:
            lime_obj = lime_obj["lime"]
        elif all(k in lime_obj for k in ["sample_index", "predicted_class", "feature_contributions", "feature_names"]):
            # 🧠 Rebuild explainer using X_train.csv
            try:
                xtrain_df = pd.read_csv(xtrain_path)
                lime_obj = LimeTabularExplainer(
                    xtrain_df.values,
                    feature_names=list(xtrain_df.columns),
                    class_names=['Low Risk (Term)', 'High Risk (PTB)'],
                    mode='classification'
                )
                print("✅ Reconstructed new LIME explainer from X_train.csv.")
            except Exception as e:
                raise RuntimeError(f"❌ Could not rebuild LIME explainer: {e}")
        else:
            raise ValueError(f"Invalid LIME object format (keys: {list(lime_obj.keys())})")

    print("✅ All assets loaded successfully.\n")
    return model, shap_bg, lime_obj


# =======================================
# === PREDICTION ========================
# =======================================
def get_prediction(model, input_df, class_names):
    probs = model.predict_proba(input_df)[0]
    idx = np.argmax(probs)
    return class_names[idx], probs[idx] * 100, probs


# =======================================
# === SHAP EXPLANATION ==================
# =======================================
def plot_shap_explanation(model, shap_bg, input_df):
    try:
        explainer = shap.TreeExplainer(model, shap_bg)
        shap_values = explainer.shap_values(input_df)
        expected = explainer.expected_value

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
            expected_val = expected[1]
        else:
            shap_vals = shap_values[0]
            expected_val = expected

        exp = shap.Explanation(
            values=shap_vals,
            base_values=expected_val,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        fig = plt.figure(figsize=(8, 6))
        shap.waterfall_plot(exp, max_display=10, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"❌ Could not generate SHAP plot: {e}")


# =======================================
# === LIME EXPLANATION ==================
# =======================================
def plot_lime_explanation(model, lime_explainer, input_df, class_names):
    try:
        def predict_fn(X):
            X_df = pd.DataFrame(X, columns=input_df.columns)
            return model.predict_proba(X_df)

        exp = lime_explainer.explain_instance(
            input_df.iloc[0].values,
            predict_fn,
            num_features=10,
            top_labels=2
        )

        fig = exp.as_pyplot_figure(label=1)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"❌ Could not generate LIME plot: {e}")
