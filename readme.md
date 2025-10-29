Preterm Birth (PTB) Risk Predictor AI App

This Streamlit web application uses a pre-trained TabPFN machine learning model to predict the risk of Preterm Birth (PTB) based on clinical and demographic data. It provides model explanations using saved SHAP and LIME explainer objects to help interpret the prediction for each patient.

Features

Single Patient Prediction: Enter a patient's information via a sidebar form to get an instant PTB risk prediction (High Risk / Low Risk) and confidence score.

AI Explanations: View interactive SHAP (waterfall) and LIME plots to understand why the model made its prediction for a specific case.

Batch Prediction: Upload a CSV file with multiple patients to receive predictions for all rows.

Download Results: Download the batch prediction results as a new CSV file.

Research Use Only: This app is intended for research purposes only and is not a substitute for professional medical advice.

Required Files

Before running the app, you must have the following three trained model and explainer files saved in the same directory as app.py and utils.py:

PTB_tabpfn_model.pkl (The core trained TabPFN model)

tabpfn_shap_results.pkl (The saved SHAP explainer/results object for feature importance)

tabpfn_lime_results.pkl (The saved LIME explainer object for local explanations)

Installation and Setup

Download the files: Ensure you have app.py, utils.py, requirements.txt, and the three .pkl files in a single project folder.

Install the required libraries:

pip install -r requirements.txt


(Note: The requirements.txt should include streamlit, pandas, numpy, joblib, shap, and matplotlib, as well as tabpfn and lime.)

Run the Application: Execute the following command in your terminal:

streamlit run app.py


Your web browser will automatically open the application.