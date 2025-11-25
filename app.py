import streamlit as st
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
import os

# ==========================================
# === FILE PATHS ===========================
# ==========================================
MODEL_PATH = "TabPFN_model.pkl"
SHAP_PATH = "TabPFN_shap_values.pkl"
LIME_PATH = "TabPFN_lime_values.pkl"
XTRAIN_PATH = "X_train.csv"

# ==========================================
# === LOGIN CREDENTIALS ====================
# ==========================================
CREDENTIALS = {"user": "predict"}  # username: user | password: predict

# ==========================================
# === PAGE CONFIG ==========================
# ==========================================
st.set_page_config(
    page_title="PTB Risk Prediction & Explainability",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# === CSS STYLING ==========================
# ==========================================
def load_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #1a1a2e;
        }
        .login-container {
            width:100%; max-width:450px; margin:150px auto;
            padding:40px; border-radius:15px;
            background: rgba(255,255,255,0.95);
            box-shadow:0 10px 30px rgba(0,0,0,0.1);
            border:1px solid #ddd;
        }
        .main-header { font-size:2rem; font-weight:800; text-align:center; color:#1a1a2e; margin-bottom:0.5rem; }
        .login-title { font-size:1.2rem; color:#4CAF50; text-align:center; margin-bottom:25px; }
        .stButton>button {
            background-color:#4CAF50; color:white; padding:10px 24px;
            border-radius:8px; border:none; transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color:#45a049; box-shadow:0 4px 12px rgba(0,0,0,0.1);
        }
        .risk-box {
            padding:20px; border-radius:12px; text-align:center;
            margin-top:20px; box-shadow:0 4px 15px rgba(0,0,0,0.1);
        }
        .risk-low { background-color:#e6ffe6; border:2px solid #4CAF50; }
        .risk-high { background-color:#ffe6e6; border:2px solid #f44336; }
        .risk-text { font-size:1.8rem; font-weight:700; margin-bottom:5px; }
        .confidence-text { font-size:1.2rem; color:#555; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# === AUTHENTICATION =======================
# ==========================================
def check_login(username, password):
    if username in CREDENTIALS and CREDENTIALS[username] == password:
        st.session_state['authenticated'] = True
    else:
        st.session_state['authenticated'] = False

def logout():
    for key in ['authenticated', 'assets_loaded', 'model', 'shap_data', 'lime_explainer']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def login_page():
    main_placeholder = st.empty()
    with main_placeholder.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown('<div style="text-align:center; color:#4CAF50; font-size:4rem; margin-bottom:5px;">🤰</div>', unsafe_allow_html=True)
            st.markdown('<div class="main-header">WELCOME TO PTB RISK PREDICTING AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-title">Secure Access Required</div>', unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("LOG IN", use_container_width=True)

                if submitted:
                    check_login(username, password)
                    if st.session_state['authenticated']:
                        main_placeholder.empty()
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center; margin-top:20px; color:#666; font-size:0.9rem;">'
                '💡 <b>DEMO Hint:</b> Username <code>user</code> | Password <code>predict</code>'
                '</p>', unsafe_allow_html=True
            )

# ==========================================
# === MAIN APPLICATION =====================
# ==========================================
def main_app(model, shap_data, lime_explainer):
    st.title("PRETERM BIRTH RISK PREDICTION DASHBOARD")
    st.sidebar.button("Logout", on_click=logout)

    feature_info = utils.get_feature_info()
    numerical_cols = feature_info['numerical_cols']
    categorical_options = feature_info['categorical_options']
    all_features = feature_info['feature_order']

    st.header("Input Patient Data")
    cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(all_features):
        with cols[i % 3]:
            display_name = feature.replace('_', ' ').title()
            if feature in numerical_cols:
                if feature == "Age":
                    user_input[feature] = st.number_input(display_name, min_value=15, max_value=50, value=25, step=1)
                elif feature == "Gravida":
                    user_input[feature] = st.number_input(display_name, min_value=1, max_value=15, value=1, step=1)
                elif feature == "ANC_followup":
                    user_input[feature] = st.slider(display_name, min_value=0, max_value=10, value=4, step=1)
                else:
                    user_input[feature] = st.number_input(display_name, value=0.0)
            else:
                options = categorical_options[feature]
                default_index = options.index("No") if "No" in options else 0
                user_input[feature] = st.selectbox(display_name, options=options, index=default_index)

    st.markdown("---")
    if st.button("🔍 PREDICT PTB RISK", use_container_width=True):
        try:
            input_df = utils.preprocess_input(user_input, feature_info)
            predicted_class, confidence, probabilities = utils.get_prediction(
                model, input_df, feature_info['class_names']
            )

            # === Display Results ===
            st.subheader("Prediction Result")
            col_result, col_probs = st.columns([2,1])
            with col_result:
                risk_class = "risk-high" if predicted_class.startswith("High") else "risk-low"
                st.markdown(f"""
                    <div class="risk-box {risk_class}">
                        <div class="risk-text">Outcome: {predicted_class}</div>
                        <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_probs:
                prob_df = pd.DataFrame(probabilities.T, index=feature_info['class_names'], columns=['Probability'])
                st.bar_chart(prob_df, use_container_width=True)

            # === Explainability ===
            st.markdown("---")
            st.header("Explainability Analysis (XAI)")
            col_shap, col_lime = st.columns(2)

            with col_shap:
                st.subheader("1. SHAP Feature Contributions")
                st.info("Explains how each feature affects the model's output.")
                try:
                    shap_fig = utils.plot_shap_explanation(model, shap_data, input_df)
                    st.pyplot(shap_fig, use_container_width=True)
                    plt.close(shap_fig)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate SHAP plot: {e}")

            with col_lime:
                st.subheader("2. LIME Local Explanation")
                st.info("Shows top features that influenced this prediction.")
                try:
                    lime_fig = utils.plot_lime_explanation(model, lime_explainer, input_df, feature_info['class_names'])
                    st.pyplot(lime_fig, use_container_width=True)
                    plt.close(lime_fig)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate LIME plot: {e}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ==========================================
# === ENTRY POINT ==========================
# ==========================================
if __name__ == "__main__":
    load_css()

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['assets_loaded'] = False

    if not st.session_state['authenticated']:
        login_page()
    else:
        if not st.session_state.get('assets_loaded', False):
            try:
                with st.spinner("Loading AI Model and Explainability Assets..."):
                    model, shap_data, lime_explainer = utils.load_all_assets(
                        MODEL_PATH, SHAP_PATH, LIME_PATH, xtrain_path=XTRAIN_PATH
                    )
                st.session_state['model'] = model
                st.session_state['shap_data'] = shap_data
                st.session_state['lime_explainer'] = lime_explainer
                st.session_state['assets_loaded'] = True
                st.success("✅ All assets loaded successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"FATAL ERROR: {e}")
                st.session_state['authenticated'] = False
                st.stop()

        if st.session_state['assets_loaded']:
            main_app(
                st.session_state['model'],
                st.session_state['shap_data'],
                st.session_state['lime_explainer']
            )





