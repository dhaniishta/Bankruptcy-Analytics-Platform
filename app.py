
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

with tab1:
    st.header(f"What-If Scenario Analysis for: **{model_choice}**")
    
    model = models[model_choice]
    prediction_proba = model.predict_proba(input_df)[0]
    bankruptcy_prob = prediction_proba[BANKRUPTCY_CLASS_INDEX]

    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Bankruptcy Risk Score")
        
        prediction_status = "High Risk" if bankruptcy_prob > 0.5 else "Low Risk"
        prediction_label = "Bankrupt" if bankruptcy_prob > 0.5 else "Non-Bankrupt"
        st.metric(label="Risk Assessment", value=prediction_status, delta=f"Prediction: {prediction_label}",
                  delta_color="inverse" if prediction_status == "Low Risk" else "normal")
        st.write("") 

        if bankruptcy_prob < 0.2: bar_color = "#2ca02c"
        elif bankruptcy_prob < 0.5: bar_color = "#ff7f0e"
        else: bar_color = "#d62728"

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = bankruptcy_prob * 100,
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
        
        baseline_pred = model.predict_proba(X_test).mean(axis=0)[BANKRUPTCY_CLASS_INDEX]
        contributions = []
        current_input = X_test.mean().to_dict()
        
        for feature in feature_names:
            previous_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][BANKRUPTCY_CLASS_INDEX]
            current_input[feature] = input_data[feature]
            new_pred = model.predict_proba(pd.DataFrame([current_input], columns=feature_names))[0][BANKRUPTCY_CLASS_INDEX]
            contributions.append(new_pred - previous_pred)

        measures = ["relative"] * len(feature_names)
        x_labels = list(feature_names.str.replace("_", " ").str.title())
        y_values = [c * 100 for c in contributions]
        
        measures = ["absolute"] + measures + ["total"]
        x_labels = ["Baseline (Avg. Company)"] + x_labels
        y_values = [baseline_pred * 100] + y_values
        
        fig_waterfall = go.Figure(go.Waterfall(
            orientation = "v", measure = measures, x = x_labels, y = y_values,
            text = [f"{v:+.2f}%" for v in y_values],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Contribution to Bankruptcy Probability (%)", showlegend=False, yaxis_title="Probability (%)")
        st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.header("Comparing Model Performance")
    
    st.subheader("Which Features Matter Most to Each Model?")
    st.info("""
    **How to Interpret This Chart:** This chart shows how much a model relies on each feature. A larger value means the feature is more important.
    *   **Why zero importance?** This is a key insight! It means the model has learned that this feature is not useful, possibly because its information is already captured by other features. Different models learn differently, so their importance rankings will vary.
    """, icon="💡")
    
    importance_model_choice = st.selectbox('Select a model to analyze its feature importance:', list(models.keys()), key="importance_model_select")
    model_to_analyze = models[importance_model_choice]

    with st.spinner(f"Analyzing feature importance for {importance_model_choice}... This may take a moment."):
        result = permutation_importance(model_to_analyze, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title=f'Feature Importance for {importance_model_choice}')
        st.plotly_chart(fig_imp, use_container_width=True)

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
    
    fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=le.inverse_transform(df['class']),
                           title="3D View of Risk Factors", labels={'color': 'Status'})
    st.plotly_chart(fig_3d, use_container_width=True)
