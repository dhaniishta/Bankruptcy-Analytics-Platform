# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import plotly.express as px
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import shap

# # Model Imports
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# # Set page configuration
# st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# # --- DATA LOADING AND PREPROCESSING ---
# @st.cache_data
# def load_data():
#     """Loads, cleans, and preprocesses the bankruptcy data."""
#     df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
#     df = df.drop_duplicates()
#     df.columns = df.columns.str.strip()
#     le = LabelEncoder()
#     df['class'] = le.fit_transform(df['class'])
#     return df, le

# # --- MODEL TRAINING ---
# @st.cache_resource
# def train_models(df):
#     """Splits data, applies SMOTE, and trains all models."""
#     X = df.drop('class', axis=1)
#     y = df['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#     models = {
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'Naive Bayes Classifier': GaussianNB(),
#         'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
#         'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
#         'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
#         'Bagging Classifier': BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
#         'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
#     }
#     trained_models = {}
#     for name, model in models.items():
#         model.fit(X_train_resampled, y_train_resampled)
#         trained_models[name] = model
#     return trained_models, X_test, y_test, X.columns, X_train_resampled

# # --- STREAMLIT UI ---
# st.title('📈 Corporate Bankruptcy Prediction App')
# st.write("""
# This application predicts whether a company will go bankrupt based on key financial and strategic risk factors. 
# It uses machine learning to analyze the provided inputs and delivers a prediction along with an explanation of the contributing factors.
# """)

# df, le = load_data()
# trained_models, X_test, y_test, feature_names, X_train_resampled = train_models(df)

# st.sidebar.header('Specify Input Parameters')
# st.sidebar.markdown("Adjust the sliders to reflect the company's risk profile.")

# def user_input_features():
#     industrial_risk = st.sidebar.select_slider('Industrial Risk', options=[0.0, 0.5, 1.0], value=0.5)
#     management_risk = st.sidebar.select_slider('Management Risk', options=[0.0, 0.5, 1.0], value=0.5)
#     financial_flexibility = st.sidebar.select_slider('Financial Flexibility', options=[0.0, 0.5, 1.0], value=0.5)
#     credibility = st.sidebar.select_slider('Credibility', options=[0.0, 0.5, 1.0], value=0.5)
#     competitiveness = st.sidebar.select_slider('Competitiveness', options=[0.0, 0.5, 1.0], value=0.5)
#     operating_risk = st.sidebar.select_slider('Operating Risk', options=[0.0, 0.5, 1.0], value=0.5)
#     model_choice = st.sidebar.selectbox('Select Prediction Model', list(trained_models.keys()))
#     data = {'industrial_risk': industrial_risk, 'management_risk': management_risk, 'financial_flexibility': financial_flexibility,
#             'credibility': credibility, 'competitiveness': competitiveness, 'operating_risk': operating_risk}
#     features = pd.DataFrame(data, index=[0])
#     return features, model_choice

# input_df, model_name = user_input_features()
# model = trained_models[model_name]

# st.subheader('User Input Parameters')
# st.dataframe(input_df, use_container_width=True)

# prediction = model.predict(input_df)
# prediction_proba = model.predict_proba(input_df)

# st.header('Prediction Result')
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader('Prediction')
#     if prediction[0] == 0:
#         st.error('**Prediction: Bankrupt**')
#     else:
#         st.success('**Prediction: Non-Bankrupt**')
# with col2:
#     st.subheader('Prediction Probability')
#     proba_df = pd.DataFrame({'Status': ['Bankrupt', 'Non-Bankrupt'], 'Probability': prediction_proba.flatten()})
#     st.dataframe(proba_df.set_index('Status'), use_container_width=True)

# # --- SHAP EXPLANATION (FINAL CORRECTED SECTION) ---
# st.header('💡 Prediction Explanation')
# st.write("""
# This plot shows how each feature contributed to the final prediction. 
# - **Features in red** push the prediction towards 'Bankrupt'.
# - **Features in blue** push the prediction towards 'Non-Bankrupt'.
# The length of the bar indicates the magnitude of the feature's impact.
# """)

# explainer = shap.Explainer(model, X_train_resampled)
# shap_explanation = explainer(input_df)

# # This robust check handles different SHAP output structures
# # If the explanation object has 3 dimensions, it means we have values for each class.
# if len(shap_explanation.shape) == 3 and shap_explanation.shape[2] > 1:
#     # We select the explanations for the "Non-Bankrupt" class (index 1)
#     plot_values = shap_explanation[0, :, 1]
# else:
#     # Otherwise, we have a single set of explanations, so we use it directly.
#     plot_values = shap_explanation[0]

# fig, ax = plt.subplots()
# shap.force_plot(plot_values, matplotlib=True, show=False, plot_cmap=['#FF0055', '#0080FF'])
# st.pyplot(fig, bbox_inches='tight')
# st.write("---")


# # --- Model Performance & Data Exploration ---
# st.header('🔍 Model Performance & Data Insights')
# with st.expander("Click to see model performance metrics and data visualizations"):
#     st.subheader(f'Performance of: {model_name}')
#     y_pred_test = model.predict(X_test)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.text('Test Set Accuracy')
#         st.info(f"{accuracy_score(y_test, y_pred_test):.2f}")
#         st.text('Classification Report (Test Set)')
#         report = classification_report(y_test, y_pred_test, target_names=['Bankrupt', 'Non-Bankrupt'], output_dict=True)
#         st.dataframe(pd.DataFrame(report).transpose())
#     with col2:
#         st.text('Confusion Matrix (Test Set)')
#         cm = confusion_matrix(y_test, y_pred_test)
#         fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
#                            x=['Bankrupt', 'Non-Bankrupt'], y=['Bankrupt', 'Non-Bankrupt'], color_continuous_scale='Blues')
#         st.plotly_chart(fig_cm)

#     if hasattr(model, 'feature_importances_'):
#         st.subheader('Feature Importance')
#         importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
#         fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
#         st.plotly_chart(fig_imp)

#     st.subheader('Exploratory Data Analysis of the Original Dataset')
#     st.text('Original Class Distribution')
#     class_dist_df = df.copy()
#     class_dist_df['class'] = class_dist_df['class'].map({0: 'Bankrupt', 1: 'Non-Bankrupt'})
#     fig_dist = px.pie(class_dist_df, names='class', title='Distribution of Bankrupt vs. Non-Bankrupt Companies', hole=0.3)
#     st.plotly_chart(fig_dist)

#     st.text('Correlation Heatmap')
#     corr = df.corr()
#     fig_corr = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap', color_continuous_scale='RdBu_r')
#     st.plotly_chart(fig_corr)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import plotly.express as px
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import shap

# # Model Imports
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# # Set page configuration
# st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# # --- DATA LOADING AND PREPROCESSING ---
# @st.cache_data
# def load_data():
#     """Loads, cleans, and preprocesses the bankruptcy data."""
#     df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
#     df = df.drop_duplicates()
#     df.columns = df.columns.str.strip()
#     le = LabelEncoder()
#     df['class'] = le.fit_transform(df['class']) # bankruptcy: 0, non-bankruptcy: 1
#     return df, le

# # --- MODEL TRAINING ---
# @st.cache_resource
# def train_models(df):
#     """Splits data, applies SMOTE, and trains all models."""
#     X = df.drop('class', axis=1)
#     y = df['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     st.write("Handling class imbalance with SMOTE...")
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#     st.write("Model training in progress...")

#     models = {
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'Naive Bayes Classifier': GaussianNB(),
#         'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
#         'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
#         'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
#         'Bagging Classifier': BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
#         'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
#     }
#     trained_models = {}
#     for name, model in models.items():
#         model.fit(X_train_resampled, y_train_resampled)
#         trained_models[name] = model
#     st.write("All models trained successfully!")
#     return trained_models, X_test, y_test, X.columns, X_train_resampled

# # --- STREAMLIT UI ---
# st.title('📈 Corporate Bankruptcy Prediction App')
# st.write("""
# This application predicts company bankruptcy based on risk factors. Adjust the sliders to see how each factor influences the outcome.
# The class imbalance in the original data (more non-bankrupt examples) has been handled using the SMOTE technique to ensure fair predictions.
# """)

# df, le = load_data()
# trained_models, X_test, y_test, feature_names, X_train_resampled = train_models(df)

# st.sidebar.header('Specify Input Parameters')
# st.sidebar.markdown("Slide to change the risk factor values.")

# def user_input_features():
#     # Using continuous sliders for more granular control
#     industrial_risk = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5, 0.01)
#     management_risk = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5, 0.01)
#     financial_flexibility = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5, 0.01)
#     credibility = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5, 0.01)
#     competitiveness = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5, 0.01)
#     operating_risk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5, 0.01)
    
#     model_choice = st.sidebar.selectbox('Select Prediction Model', list(trained_models.keys()))

#     data = {'industrial_risk': industrial_risk, 'management_risk': management_risk, 'financial_flexibility': financial_flexibility,
#             'credibility': credibility, 'competitiveness': competitiveness, 'operating_risk': operating_risk}
#     features = pd.DataFrame(data, index=[0])
#     return features, model_choice

# input_df, model_name = user_input_features()
# model = trained_models[model_name]

# st.subheader('User Input Parameters')
# st.dataframe(input_df, use_container_width=True)

# prediction = model.predict(input_df)
# prediction_proba = model.predict_proba(input_df)

# st.header('Prediction Result')
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader('Prediction')
#     if prediction[0] == 0:
#         st.error('**Prediction: Bankrupt**')
#     else:
#         st.success('**Prediction: Non-Bankrupt**')
# with col2:
#     st.subheader('Prediction Probability')
#     proba_df = pd.DataFrame({'Status': ['Bankrupt', 'Non-Bankrupt'], 'Probability': prediction_proba.flatten()})
#     st.dataframe(proba_df.set_index('Status'), use_container_width=True)

# # --- SHAP EXPLANATION (FINAL ROBUST VERSION) ---
# st.header('💡 Prediction Explanation')
# st.write("""
# This plot shows how each feature contributed to the final prediction. Red features push towards bankruptcy, blue features push away from it.
# """)

# # We need to use different explainers for different model types.
# # Tree-based models are faster. Others need the KernelExplainer.
# is_tree_model = hasattr(model, 'feature_importances_')

# if is_tree_model:
#     explainer = shap.Explainer(model, X_train_resampled)
#     shap_values = explainer(input_df)
# else:
#     # For non-tree models, we use KernelExplainer which needs a prediction function and background data.
#     # We summarize the background data with shap.kmeans to make it faster.
#     background_data_summary = shap.kmeans(X_train_resampled, 10)
#     explainer = shap.KernelExplainer(model.predict_proba, background_data_summary)
#     shap_values = explainer.shap_values(input_df)

# # Now, we correctly extract the values for plotting
# # The structure of shap_values is different for KernelExplainer (list) vs. TreeExplainer (Explanation object)
# if isinstance(shap_values, list):
#     # For KernelExplainer, shap_values is a list [class_0_values, class_1_values]
#     # We plot the explanation for the 'Non-Bankrupt' class (index 1)
#     plot_values = shap_values[1]
#     base_value = explainer.expected_value[1]
# else:
#     # For TreeExplainer, it's a multi-output object. We select class 1.
#     plot_values = shap_values[0, :, 1].values
#     base_value = shap_values.base_values[0, 1]

# fig, ax = plt.subplots()
# shap.force_plot(base_value, plot_values, input_df, matplotlib=True, show=False, plot_cmap=['#FF0055', '#0080FF'])
# st.pyplot(fig, bbox_inches='tight', clear_figure=True)
# st.write("---")

# # --- Model Performance & Data Exploration ---
# st.header('🔍 Model Performance & Data Insights')
# with st.expander("Click to see model performance metrics and data visualizations"):
#     st.subheader(f'Performance of: {model_name}')
#     y_pred_test = model.predict(X_test)
#     # ... (Rest of the expander code is the same) ...
#     col1, col2 = st.columns(2)
#     with col1:
#         st.text('Test Set Accuracy')
#         st.info(f"{accuracy_score(y_test, y_pred_test):.2f}")
#         st.text('Classification Report (Test Set)')
#         report = classification_report(y_test, y_pred_test, target_names=['Bankrupt', 'Non-Bankrupt'], output_dict=True)
#         st.dataframe(pd.DataFrame(report).transpose())
#     with col2:
#         st.text('Confusion Matrix (Test Set)')
#         cm = confusion_matrix(y_test, y_pred_test)
#         fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
#                            x=['Bankrupt', 'Non-Bankrupt'], y=['Bankrupt', 'Non-Bankrupt'], color_continuous_scale='Blues')
#         st.plotly_chart(fig_cm)

#     if is_tree_model:
#         st.subheader('Feature Importance')
#         importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
#         fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (Overall)')
#         st.plotly_chart(fig_imp)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import shap

# # Model Imports
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# # Set page configuration
# st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# # --- DATA LOADING AND PREPROCESSING ---
# @st.cache_data
# def load_data():
#     """Loads, cleans, and preprocesses the bankruptcy data."""
#     df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
#     df = df.drop_duplicates()
#     df.columns = df.columns.str.strip()
#     le = LabelEncoder()
#     df['class'] = le.fit_transform(df['class']) # bankruptcy: 0, non-bankruptcy: 1
#     return df, le

# # --- MODEL TRAINING ---
# @st.cache_resource
# def train_models(df):
#     """Splits data, applies SMOTE, and trains all models."""
#     X = df.drop('class', axis=1)
#     y = df['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#     models = {
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
#         'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
#         'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
#         'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
#         'Naive Bayes Classifier': GaussianNB(),
#         'Bagging Classifier': BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
#     }
#     trained_models = {}
#     for name, model in models.items():
#         model.fit(X_train_resampled, y_train_resampled)
#         trained_models[name] = model
        
#     return trained_models, X_test, y_test, X.columns, X_train_resampled

# # --- STREAMLIT UI ---
# st.title('📈 Corporate Bankruptcy Prediction App')
# st.write("""
# This application predicts company bankruptcy based on risk factors. Adjust the sliders to see how each factor influences the outcome.
# The class imbalance in the original data (more non-bankrupt examples) has been handled using the **SMOTE** technique to ensure fair predictions.
# """)

# df, le = load_data()
# trained_models, X_test, y_test, feature_names, X_train_resampled = train_models(df)

# st.sidebar.header('Specify Input Parameters')
# st.sidebar.markdown("Slide to change the risk factor values.")

# def user_input_features():
#     industrial_risk = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5, 0.01)
#     management_risk = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5, 0.01)
#     financial_flexibility = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5, 0.01)
#     credibility = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5, 0.01)
#     competitiveness = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5, 0.01)
#     operating_risk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5, 0.01)
#     model_choice = st.sidebar.selectbox('Select Prediction Model', list(trained_models.keys()))
#     data = {'industrial_risk': industrial_risk, 'management_risk': management_risk, 'financial_flexibility': financial_flexibility,
#             'credibility': credibility, 'competitiveness': competitiveness, 'operating_risk': operating_risk}
#     features = pd.DataFrame(data, index=[0])
#     return features, model_choice

# input_df, model_name = user_input_features()
# model = trained_models[model_name]

# st.subheader('User Input Parameters')
# st.dataframe(input_df, use_container_width=True)

# prediction = model.predict(input_df)
# prediction_proba = model.predict_proba(input_df)

# st.header('Prediction Result')
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader('Prediction')
#     if prediction[0] == 0:
#         st.error('**Prediction: Bankrupt**')
#     else:
#         st.success('**Prediction: Non-Bankrupt**')
# with col2:
#     st.subheader('Prediction Probability')
#     proba_df = pd.DataFrame({'Status': ['Bankrupt', 'Non-Bankrupt'], 'Probability': prediction_proba.flatten()})
#     st.dataframe(proba_df.set_index('Status'), use_container_width=True)

# # --- SHAP EXPLANATION (FINAL ROBUST VERSION) ---
# st.header('💡 Prediction Explanation')
# st.write("""
# This plot shows which features are pushing the prediction towards bankruptcy (red) or non-bankruptcy (blue). The larger the bar, the greater the feature's impact.
# """)

# try:
#     # This robust `try...except` block ensures compatibility with all models.
#     try:
#         # First, try the fast TreeExplainer, best for tree-based models.
#         # We use .shap_values() which gives a consistent list-of-arrays output.
#         explainer = shap.TreeExplainer(model, X_train_resampled)
#         shap_values_list = explainer.shap_values(input_df)
#         base_value = explainer.expected_value[1] # Base value for the 'Non-Bankrupt' class

#     except Exception:
#         # If TreeExplainer fails, fall back to KernelExplainer. It's slower but works for any model.
#         st.info(f"Using KernelExplainer for {model_name}. This may take a moment...")
#         background_data_summary = shap.kmeans(X_train_resampled, 10)
#         explainer = shap.KernelExplainer(model.predict_proba, background_data_summary)
#         shap_values_list = explainer.shap_values(input_df)
#         base_value = explainer.expected_value[1]

#     # Both methods produce a list [class_0_values, class_1_values]. We want the second one.
#     # The shape is (num_samples, num_features). We only have 1 sample, so we take index [0].
#     plot_values = shap_values_list[1][0]

#     # Create and display the plot
#     fig, ax = plt.subplots()
#     shap.force_plot(base_value, plot_values, input_df.iloc[0], matplotlib=True, show=False, plot_cmap=['#FF0055', '#0080FF'])
#     st.pyplot(fig, bbox_inches='tight', clear_figure=True)

# except Exception as e:
#     st.error(f"Could not generate SHAP plot for the selected model ({model_name}).")
#     st.error(f"Error: {e}")

# st.write("---")

# # --- Model Performance & Data Exploration ---
# st.header('🔍 Model Performance & Data Insights')
# with st.expander("Click to see model performance metrics and data visualizations"):
#     st.subheader(f'Performance of: {model_name}')
#     y_pred_test = model.predict(X_test)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.text('Test Set Accuracy')
#         st.info(f"{accuracy_score(y_test, y_pred_test):.2f}")
#         st.text('Classification Report (Test Set)')
#         report = classification_report(y_test, y_pred_test, target_names=['Bankrupt', 'Non-Bankrupt'], output_dict=True)
#         st.dataframe(pd.DataFrame(report).transpose())
#     with col2:
#         st.text('Confusion Matrix (Test Set)')
#         cm = confusion_matrix(y_test, y_pred_test)
#         fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
#                            x=['Bankrupt', 'Non-Bankrupt'], y=['Bankrupt', 'Non-Bankrupt'], color_continuous_scale='Blues')
#         st.plotly_chart(fig_cm)

#     if hasattr(model, 'feature_importances_'):
#         st.subheader('Feature Importance')
#         importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
#         fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (Overall)')
#         st.plotly_chart(fig_imp)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.inspection import permutation_importance

# # --- MODEL IMPORTS ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# # --- APP CONFIGURATION ---
# st.set_page_config(page_title="Bankruptcy Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# # --- DATA LOADING AND MODEL TRAINING (CACHED) ---
# @st.cache_data
# def load_data():
#     df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
#     df = df.drop_duplicates()
#     df.columns = df.columns.str.strip()
#     le = LabelEncoder()
#     # Store the mapping for later use
#     class_mapping = {label: i for i, label in enumerate(le.fit(df['class']).classes_)}
#     df['class'] = le.transform(df['class']) # bankruptcy: 0, non-bankruptcy: 1
#     return df, le, class_mapping

# @st.cache_resource
# def train_models(df):
#     X = df.drop('class', axis=1)
#     y = df['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
#         'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
#         'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
#         'Bagging Classifier': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
#         'Naive Bayes Classifier': GaussianNB(),
#     }
#     for name, model in models.items():
#         model.fit(X_train_resampled, y_train_resampled)
#     return models, X_test, y_test, X.columns

# # --- LOAD DATA ---
# df, le, class_mapping = load_data()
# models, X_test, y_test, feature_names = train_models(df)

# # --- SIDEBAR - USER INPUTS ---
# st.sidebar.title("🎛️ Control Panel")
# st.sidebar.header("Scenario Parameters")
# model_choice = st.sidebar.selectbox('Select Prediction Model', list(models.keys()))

# input_data = {}
# for feature in feature_names:
#     input_data[feature] = st.sidebar.slider(feature.replace("_", " ").title(), 0.0, 1.0, 0.5, 0.01)
# input_df = pd.DataFrame([input_data])

# # --- MAIN APP ---
# st.title("📊 Interactive Bankruptcy Analytics Dashboard")
# st.markdown("An advanced tool for exploring bankruptcy risk factors, analyzing predictive models, and running 'what-if' scenarios.")

# # --- UI TABS ---
# tab1, tab2, tab3 = st.tabs(["🔮 **Prediction Playground**", "📊 **Model Deep Dive**", "📈 **Data Explorer**"])

# # =================================================================================================
# # TAB 1: PREDICTION PLAYGROUND
# # =================================================================================================
# with tab1:
#     st.header(f"What-If Scenario Analysis for: **{model_choice}**")
    
#     model = models[model_choice]
#     prediction_proba = model.predict_proba(input_df)[0]
#     bankruptcy_prob = prediction_proba[class_mapping['bankruptcy']]

#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("Bankruptcy Risk Score")
        
#         # --- GAUGE CHART ---
#         fig_gauge = go.Figure(go.Indicator(
#             mode = "gauge+number",
#             value = bankruptcy_prob * 100,
#             number = {'suffix': '%'},
#             title = {'text': "Probability of Bankruptcy"},
#             gauge = {'axis': {'range': [None, 100]},
#                      'steps' : [
#                          {'range': [0, 20], 'color': "green"},
#                          {'range': [20, 50], 'color': "orange"},
#                          {'range': [50, 100], 'color': "red"}],
#                      'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
#         st.plotly_chart(fig_gauge, use_container_width=True)

#         if bankruptcy_prob > 0.5:
#             st.error(f"**High Risk of Bankruptcy** (Prediction: Bankrupt)")
#         else:
#             st.success(f"**Low Risk of Bankruptcy** (Prediction: Non-Bankrupt)")

#     with col2:
#         st.subheader("How Each Factor Influences the Score")
#         st.markdown("This waterfall chart shows how each risk factor pushes the bankruptcy probability up or down from the average case.")
        
#         # --- WATERFALL CHART CALCULATION ---
#         baseline_pred = model.predict_proba(pd.DataFrame([X_test.mean()], columns=feature_names))[0][class_mapping['bankruptcy']]
        
#         contributions = []
#         current_input = X_test.mean().to_dict()
        
#         for feature in feature_names:
#             previous_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][class_mapping['bankruptcy']]
#             current_input[feature] = input_data[feature]
#             new_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][class_mapping['bankruptcy']]
#             contributions.append(new_pred - previous_pred)
            
#         fig_waterfall = go.Figure(go.Waterfall(
#             name = "Prediction Breakdown", 
#             orientation = "v",
#             measure = ["relative"] * len(feature_names) + ["total"],
#             x = list(feature_names.str.replace("_", " ").str.title()) + ["Final Score"],
#             y = contributions + [sum(contributions)],
#             text = [f"{c*100:+.1f}%" for c in contributions] + [f"{sum(contributions)*100:+.1f}%"],
#             base = baseline_pred,
#             connector = {"line":{"color":"rgb(63, 63, 63)"}},
#         ))
#         fig_waterfall.update_layout(title="Contribution to Bankruptcy Probability", showlegend=False)
#         st.plotly_chart(fig_waterfall, use_container_width=True)


# # =================================================================================================
# # TAB 2: MODEL DEEP DIVE
# # =================================================================================================
# with tab2:
#     st.header("Comparing Model Performance")
    
#     # --- PERMUTATION FEATURE IMPORTANCE ---
#     st.subheader("Which Features Matter Most to Each Model?")
#     st.markdown("Permutation Importance measures how much a model's performance decreases when a feature's values are randomly shuffled. A larger drop means the feature is more important.")
    
#     importance_model_choice = st.selectbox('Select a model to analyze its feature importance:', list(models.keys()))
#     model_to_analyze = models[importance_model_choice]

#     with st.spinner("Calculating permutation importance..."):
#         result = permutation_importance(model_to_analyze, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
#         importance_df = pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=True)
        
#         fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title=f'Feature Importance for {importance_model_choice}')
#         st.plotly_chart(fig_imp, use_container_width=True)

# # =================================================================================================
# # TAB 3: DATA EXPLORER
# # =================================================================================================
# with tab3:
#     st.header("Exploring the Dataset")
    
#     # --- CLASS DISTRIBUTION ---
#     st.subheader("Original Class Distribution")
#     fig_pie = px.pie(df, names=le.inverse_transform(df['class']), title="Bankruptcy vs. Non-Bankruptcy Companies", hole=0.3)
#     st.plotly_chart(fig_pie, use_container_width=True)
    
#     # --- 3D SCATTER PLOT ---
#     st.subheader("Interactive 3D Scatter Plot")
#     st.markdown("Explore the relationships between any three features.")
    
#     col1, col2, col3 = st.columns(3)
#     x_axis = col1.selectbox("X-Axis", feature_names, index=0)
#     y_axis = col2.selectbox("Y-Axis", feature_names, index=1)
#     z_axis = col3.selectbox("Z-Axis", feature_names, index=2)
    
#     fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=le.inverse_transform(df['class']),
#                            title="3D View of Risk Factors", labels={'color': 'Status'})
#     st.plotly_chart(fig_3d, use_container_width=True)
    
#     # --- PARALLEL COORDINATES PLOT ---
#     st.subheader("Parallel Coordinates Plot")
#     st.markdown("Each line is a company. This plot helps visualize how risk profiles differ between bankrupt and non-bankrupt companies across all factors simultaneously.")
    
#     fig_par = go.Figure(data=
#         go.Parcoords(
#             line = dict(color = df['class'],
#                        colorscale = [[0,'red'],[1,'green']]),
#             dimensions = list([
#                 dict(range = [0,1], label = col.replace("_", " ").title(), values = df[col]) for col in feature_names
#             ])
#         )
#     )
#     st.plotly_chart(fig_par, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.inspection import permutation_importance

# # --- MODEL IMPORTS ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# # --- APP CONFIGURATION ---
# st.set_page_config(page_title="Bankruptcy Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# # --- DATA LOADING AND MODEL TRAINING (CACHED) ---
# @st.cache_data
# def load_data():
#     df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
#     df = df.drop_duplicates()
#     df.columns = df.columns.str.strip()
#     le = LabelEncoder()
#     # Store the mapping for later use
#     class_mapping = {label: i for i, label in enumerate(le.fit(df['class']).classes_)}
#     df['class'] = le.transform(df['class']) # bankruptcy: 0, non-bankruptcy: 1
#     return df, le, class_mapping

# @st.cache_resource
# def train_models(df):
#     X = df.drop('class', axis=1)
#     y = df['class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
#         'Logistic Regression': LogisticRegression(random_state=42),
#         'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
#         'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
#         'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
#         'Bagging Classifier': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
#         'Naive Bayes Classifier': GaussianNB(),
#     }
#     for name, model in models.items():
#         model.fit(X_train_resampled, y_train_resampled)
#     return models, X_test, y_test, X.columns

# # --- LOAD DATA ---
# df, le, class_mapping = load_data()
# models, X_test, y_test, feature_names = train_models(df)

# # --- SIDEBAR - USER INPUTS ---
# st.sidebar.title("🎛️ Control Panel")
# st.sidebar.header("Scenario Parameters")
# model_choice = st.sidebar.selectbox('Select Prediction Model', list(models.keys()))

# input_data = {}
# for feature in feature_names:
#     input_data[feature] = st.sidebar.slider(feature.replace("_", " ").title(), 0.0, 1.0, 0.5, 0.01)
# input_df = pd.DataFrame([input_data])

# # --- MAIN APP ---
# st.title("📊 Interactive Bankruptcy Analytics Dashboard")
# st.markdown("An advanced tool for exploring bankruptcy risk factors, analyzing predictive models, and running 'what-if' scenarios.")

# # --- UI TABS ---
# tab1, tab2, tab3 = st.tabs(["🔮 **Prediction Playground**", "📊 **Model Deep Dive**", "📈 **Data Explorer**"])

# # =================================================================================================
# # TAB 1: PREDICTION PLAYGROUND
# # =================================================================================================
# with tab1:
#     st.header(f"What-If Scenario Analysis for: **{model_choice}**")
    
#     model = models[model_choice]
#     prediction_proba = model.predict_proba(input_df)[0]
#     bankruptcy_prob = prediction_proba[class_mapping['bankruptcy']]

#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("Bankruptcy Risk Score")
        
#         # --- GAUGE CHART ---
#         fig_gauge = go.Figure(go.Indicator(
#             mode = "gauge+number",
#             value = bankruptcy_prob * 100,
#             number = {'suffix': '%'},
#             title = {'text': "Probability of Bankruptcy"},
#             gauge = {'axis': {'range': [None, 100]},
#                      'steps' : [
#                          {'range': [0, 20], 'color': "green"},
#                          {'range': [20, 50], 'color': "orange"},
#                          {'range': [50, 100], 'color': "red"}],
#                      'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
#         st.plotly_chart(fig_gauge, use_container_width=True)

#         if bankruptcy_prob > 0.5:
#             st.error(f"**High Risk of Bankruptcy** (Prediction: Bankrupt)")
#         else:
#             st.success(f"**Low Risk of Bankruptcy** (Prediction: Non-Bankrupt)")

#     with col2:
#         st.subheader("How Each Factor Influences the Score")
#         st.markdown("This waterfall chart shows how each risk factor pushes the bankruptcy probability up or down from the average case.")
        
#         # --- WATERFALL CHART CALCULATION ---
#         baseline_pred = model.predict_proba(pd.DataFrame([X_test.mean()], columns=feature_names))[0][class_mapping['bankruptcy']]
        
#         contributions = []
#         current_input = X_test.mean().to_dict()
        
#         for feature in feature_names:
#             previous_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][class_mapping['bankruptcy']]
#             current_input[feature] = input_data[feature]
#             new_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][class_mapping['bankruptcy']]
#             contributions.append(new_pred - previous_pred)
            
#         fig_waterfall = go.Figure(go.Waterfall(
#             name = "Prediction Breakdown", 
#             orientation = "v",
#             measure = ["relative"] * len(feature_names) + ["total"],
#             x = list(feature_names.str.replace("_", " ").str.title()) + ["Final Score"],
#             y = contributions + [sum(contributions)],
#             text = [f"{c*100:+.1f}%" for c in contributions] + [f"{sum(contributions)*100:+.1f}%"],
#             base = baseline_pred,
#             connector = {"line":{"color":"rgb(63, 63, 63)"}},
#         ))
#         fig_waterfall.update_layout(title="Contribution to Bankruptcy Probability", showlegend=False)
#         st.plotly_chart(fig_waterfall, use_container_width=True)


# # =================================================================================================
# # TAB 2: MODEL DEEP DIVE
# # =================================================================================================
# with tab2:
#     st.header("Comparing Model Performance")
    
#     # --- PERMUTATION FEATURE IMPORTANCE ---
#     st.subheader("Which Features Matter Most to Each Model?")
#     st.markdown("Permutation Importance measures how much a model's performance decreases when a feature's values are randomly shuffled. A larger drop means the feature is more important.")
    
#     importance_model_choice = st.selectbox('Select a model to analyze its feature importance:', list(models.keys()))
#     model_to_analyze = models[importance_model_choice]

#     with st.spinner("Calculating permutation importance..."):
#         result = permutation_importance(model_to_analyze, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
#         importance_df = pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=True)
        
#         fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title=f'Feature Importance for {importance_model_choice}')
#         st.plotly_chart(fig_imp, use_container_width=True)

# # =================================================================================================
# # TAB 3: DATA EXPLORER
# # =================================================================================================
# with tab3:
#     st.header("Exploring the Dataset")
    
#     # --- CLASS DISTRIBUTION ---
#     st.subheader("Original Class Distribution")
#     fig_pie = px.pie(df, names=le.inverse_transform(df['class']), title="Bankruptcy vs. Non-Bankruptcy Companies", hole=0.3)
#     st.plotly_chart(fig_pie, use_container_width=True)
    
#     # --- 3D SCATTER PLOT ---
#     st.subheader("Interactive 3D Scatter Plot")
#     st.markdown("Explore the relationships between any three features.")
    
#     col1, col2, col3 = st.columns(3)
#     x_axis = col1.selectbox("X-Axis", feature_names, index=0)
#     y_axis = col2.selectbox("Y-Axis", feature_names, index=1)
#     z_axis = col3.selectbox("Z-Axis", feature_names, index=2)
    
#     fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=le.inverse_transform(df['class']),
#                            title="3D View of Risk Factors", labels={'color': 'Status'})
#     st.plotly_chart(fig_3d, use_container_width=True)
    
#     # --- PARALLEL COORDINATES PLOT ---
#     st.subheader("Parallel Coordinates Plot")
#     st.markdown("Each line is a company. This plot helps visualize how risk profiles differ between bankrupt and non-bankrupt companies across all factors simultaneously.")
    
#     fig_par = go.Figure(data=
#         go.Parcoords(
#             line = dict(color = df['class'],
#                        colorscale = [[0,'red'],[1,'green']]),
#             dimensions = list([
#                 dict(range = [0,1], label = col.replace("_", " ").title(), values = df[col]) for col in feature_names
#             ])
#         )
#     )
#     st.plotly_chart(fig_par, use_container_width=True)



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# --- MODEL IMPORTS ---
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Bankruptcy Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING AND MODEL TRAINING (CACHED) ---
@st.cache_data
def load_data():
    df = pd.read_csv('bankruptcy-prevention.csv', sep=";")
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    le = LabelEncoder()
    class_mapping = {label: i for i, label in enumerate(le.fit(df['class']).classes_)}
    df['class'] = le.transform(df['class'])
    return df, le, class_mapping

@st.cache_resource
def train_models(df):
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'SVC': SVC(kernel='poly', degree=6, probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Bagging Classifier': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
        'Naive Bayes Classifier': GaussianNB(),
    }
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
    return models, X_test, y_test, X.columns

# --- LOAD DATA ---
df, le, class_mapping = load_data()
models, X_test, y_test, feature_names = train_models(df)
BANKRUPTCY_CLASS_INDEX = class_mapping.get('bankruptcy', 0)

# --- SIDEBAR - USER INPUTS ---
st.sidebar.title("🎛️ Control Panel")
st.sidebar.header("Scenario Parameters")
model_choice = st.sidebar.selectbox('Select Prediction Model', list(models.keys()))

input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.slider(feature.replace("_", " ").title(), 0.0, 1.0, 0.5, 0.01)
input_df = pd.DataFrame([input_data])

# --- MAIN APP ---
st.title("📊 Interactive Bankruptcy Analytics Dashboard")
st.markdown("An advanced tool for exploring bankruptcy risk factors, analyzing predictive models, and running 'what-if' scenarios.")

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 **Prediction Playground**", "📊 **Model Deep Dive**", "📈 **Data Explorer**"])

# =================================================================================================
# TAB 1: PREDICTION PLAYGROUND
# =================================================================================================
with tab1:
    st.header(f"What-If Scenario Analysis for: **{model_choice}**")
    
    model = models[model_choice]
    prediction_proba = model.predict_proba(input_df)[0]
    bankruptcy_prob = prediction_proba[BANKRUPTCY_CLASS_INDEX]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bankruptcy Risk Score")
        
        # --- KPI BOX FOR PREDICTION (MOVED UP) ---
        prediction_status = "High Risk" if bankruptcy_prob > 0.5 else "Low Risk"
        prediction_label = "Bankrupt" if bankruptcy_prob > 0.5 else "Non-Bankrupt"
        st.metric(label="Risk Assessment", value=prediction_status, delta=f"Prediction: {prediction_label}",
                  delta_color="inverse" if prediction_status == "Low Risk" else "normal")
        st.write("") # Add a little space

        # --- DYNAMIC GAUGE CHART (IMPROVED) ---
        if bankruptcy_prob < 0.2: bar_color = "#2ca02c" # Bold Green
        elif bankruptcy_prob < 0.5: bar_color = "#ff7f0e" # Bold Orange
        else: bar_color = "#d62728" # Bold Red

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bankruptcy_prob * 100,
            number = {'suffix': '%', 'font': {'size': 50}},
            title = {'text': "Probability of Bankruptcy", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'steps' : [{'range': [0, 20], 'color': "#d2f0d2"}, {'range': [20, 50], 'color': "#f0e4c7"}, {'range': [50, 100], 'color': "#f0c7c7"}],
                'bar': {'color': bar_color, 'thickness': 0.3},
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': 50}
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=80, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("How Each Factor Influences the Score")
        st.markdown("This waterfall chart shows how each factor contributes to the final score, starting from an average baseline case.")
        
        # --- WATERFALL CHART (IMPROVED LOGIC) ---
        baseline_pred = model.predict_proba(X_test).mean(axis=0)[BANKRUPTCY_CLASS_INDEX]
        
        contributions = []
        current_input = X_test.mean().to_dict()
        
        for feature in feature_names:
            previous_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][BANKRUPTCY_CLASS_INDEX]
            current_input[feature] = input_data[feature]
            new_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][BANKRUPTCY_CLASS_INDEX]
            contributions.append(new_pred - previous_pred)

        # Create measures for the waterfall chart
        measures = ["relative"] * len(feature_names)
        x_labels = list(feature_names.str.replace("_", " ").str.title())
        y_values = [c * 100 for c in contributions]
        
        # Add the initial baseline and the final total
        measures = ["absolute"] + measures + ["total"]
        x_labels = ["Baseline (Avg. Company)"] + x_labels
        y_values = [baseline_pred * 100] + y_values
        
        fig_waterfall = go.Figure(go.Waterfall(
            orientation = "v",
            measure = measures,
            x = x_labels,
            y = y_values,
            text = [f"{v:+.2f}%" for v in y_values],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Contribution to Bankruptcy Probability (%)", showlegend=False, yaxis_title="Probability (%)")
        st.plotly_chart(fig_waterfall, use_container_width=True)

# =================================================================================================
# TAB 2: MODEL DEEP DIVE
# =================================================================================================
with tab2:
    st.header("Comparing Model Performance")
    
    st.subheader("Which Features Matter Most to Each Model?")
    importance_model_choice = st.selectbox('Select a model to analyze its feature importance:', list(models.keys()), key="importance_model_select")
    model_to_analyze = models[importance_model_choice]

    with st.spinner(f"Analyzing feature importance for {importance_model_choice}... This may take a moment."):
        result = permutation_importance(model_to_analyze, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title=f'Feature Importance for {importance_model_choice}')
        st.plotly_chart(fig_imp, use_container_width=True)

# =================================================================================================
# TAB 3: DATA EXPLORER
# =================================================================================================
with tab3:
    st.header("Exploring the Dataset")
    
    col1_exp, col2_exp = st.columns(2)
    with col1_exp:
        st.subheader("Original Class Distribution")
        fig_pie = px.pie(df, names=le.inverse_transform(df['class']), title="Bankruptcy vs. Non-Bankruptcy Companies", hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2_exp:
        st.subheader("Parallel Coordinates Plot")
        st.markdown("Each line is a company. This plot helps visualize how risk profiles differ between bankrupt (red) and non-bankrupt (green) companies.")
        fig_par = go.Figure(data=
            go.Parcoords(
                line = dict(color = df['class'], colorscale = [[0,'red'],[1,'green']]),
                dimensions = list([
                    dict(range = [0,1], label = col.replace("_", " ").title(), values = df[col]) for col in feature_names
                ])
            )
        )
        st.plotly_chart(fig_par, use_container_width=True)
    
    st.subheader("Interactive 3D Scatter Plot")
    st.markdown("Explore the relationships between any three features.")
    
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    x_axis = plot_col1.selectbox("X-Axis", feature_names, index=0)
    y_axis = plot_col2.selectbox("Y-Axis", feature_names, index=1)
    z_axis = plot_col3.selectbox("Z-Axis", feature_names, index=2)
    
    # Corrected function name from scatter_d to scatter_3d
    fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=le.inverse_transform(df['class']),
                           title="3D View of Risk Factors", labels={'color': 'Status'})
    st.plotly_chart(fig_3d, use_container_width=True)