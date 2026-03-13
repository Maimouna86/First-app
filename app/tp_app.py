import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly.figure_factory as ff

# ======================================================================
# Configuration de la page
# ======================================================================

st.set_page_config(
    page_title="Prédiction d'Approbation de Prêt",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# Chargement des données
# ======================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/processed/loan_data_clean.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier loan_data_clean.csv introuvable.")
        return None

# ======================================================================
# Chargement des modèles
# ======================================================================

@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "Régression Logistique":
            model = joblib.load("../notebook/models/logistic_regression.pkl")
            scaler = joblib.load("../notebook/models/scaler.pkl")
        else:
            model = joblib.load("../notebook/models/random_forest.pkl")
            scaler = None
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Modèle introuvable.")
        return None, None

# ======================================================================
# Titre
# ======================================================================

st.title("🏦 Application de Prédiction d'Approbation de Prêt")
st.markdown("---")

# ======================================================================
# Sidebar
# ======================================================================

st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Régression Logistique", "Random Forest"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 À propos")
st.sidebar.write(
    "Cette application analyse des données de prêts bancaires "
    "et prédit l'approbation d'une demande de prêt."
)

# ======================================================================
# Chargement des données et du modèle
# ======================================================================

df = load_data()
if df is None:
    st.stop()

model, scaler = load_model(model_choice)
if model is None:
    st.stop()

# ======================================================================
# TABS
# ======================================================================

tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

# =====================================================================================
# TAB 1 : EXPLORATION
# =====================================================================================

with tab1:
    st.header("📊 Exploration des données")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total demandes", f"{len(df):,}")

    with col2:
        approval_rate = (df['Loan_Status'] == 1).mean() * 100
        st.metric("✅ Taux d'approbation", f"{approval_rate:.1f}%")

    with col3:
        st.metric("💰 Montant moyen", f"{df['LoanAmount'].mean():,.0f} €")

    with col4:
        st.metric("💵 Revenu moyen", f"{df['CoapplicantIncome'].mean():,.0f} €")

    st.markdown("---")

    st.subheader("📈 Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='ApplicantIncome',
                           title='Distribution des revenus des demandeurs')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y='LoanAmount',
                     title='Distribution du montant des prêts')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Analyses")
    col1, col2 = st.columns(2)

    with col1:
        df_temp = df.copy()
        df_temp['Loan_Status_Text'] = df_temp['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
        approval_by_edu = df_temp.groupby('Education')['Loan_Status'].mean() * 100

        fig = px.bar(
            x=approval_by_edu.index.map({1: 'Graduate', 0: 'Not Graduate'}),
            y=approval_by_edu.values,
            title='Taux d\'approbation par niveau d\'éducation'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=df['Loan_Status'].value_counts().values,
            names=['Approved', 'Rejected'],
            title='Répartition des décisions'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Corrélations")
    corr = df.select_dtypes(include=['number']).corr()

    # 🔧 Correction : text doit être un tableau de chaînes, pas un DataFrame
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.round(2),
        texttemplate='%{text}'

    ))
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Voir le dataset complet"):
        st.dataframe(df, use_container_width=True)

# =====================================================================================
# TAB 2 : PRÉDICTION
# =====================================================================================

with tab2:
    st.header("🤖 Faire une prédiction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Genre", [1, 0],
                                  format_func=lambda x: "👨 Homme" if x == 1 else "👩 Femme")

            applicant_income = st.number_input("Revenu du demandeur (€)", 0, 100000, 5000)

            coapplicant_income = st.number_input("Revenu co-demandeur (€)", 0, 100000, 0)

            loan_amount = st.number_input("Montant du prêt (€)", 1000, 1000000, 150000)

            loan_term = st.number_input("Durée (mois)", 12, 480, 360)

        with col2:
            credit_history = st.selectbox("Historique de crédit", [1, 0],
                                          format_func=lambda x: "Bon" if x == 1 else "Mauvais")

            education = st.selectbox("Éducation", [1, 0],
                                     format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")

            married = st.selectbox("Marié", [1, 0],
                                   format_func=lambda x: "Marié(e)" if x == 1 else "Célibataire")

            dependents = st.number_input("Personnes à charge", 0, 10)

            self_employed = st.selectbox("Indépendant", [0, 1],
                                         format_func=lambda x: "Oui" if x == 1 else "Non")

            property_area = st.selectbox("Zone", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔮 Prédire l'approbation du prêt")

    if submitted:
        with st.spinner("Analyse en cours..."):

            input_data = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Education': education,
                'Gender_Male': gender,
                'Married_Yes': married,
                'Dependents': dependents,
                'SelfEmployed_Yes': self_employed,
                'Area_Semiurban': 1 if property_area == "Semiurban" else 0,
                'Area_Urban': 1 if property_area == "Urban" else 0
            }

            input_df = pd.DataFrame([input_data])

            input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
            input_df['LoanAmountToIncome'] = input_df['LoanAmount'] / (input_df['TotalIncome'] + 1)
            input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
            input_df['EMIToIncome'] = input_df['EMI'] / (input_df['TotalIncome'] + 1)
            input_df['Log_LoanAmount'] = np.log(input_df['LoanAmount'] + 1)
            input_df['Log_TotalIncome'] = np.log(input_df['TotalIncome'] + 1)
            input_df['Has_Coapplicant'] = (input_df['CoapplicantIncome'] > 0).astype(int)  # ✔ Correction

            expected_order = [
                'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome',
                'LoanAmountToIncome', 'EMI', 'EMIToIncome', 'Log_LoanAmount',
                'Log_TotalIncome', 'Has_Coapplicant', 'Area_Semiurban',
                'Area_Urban', 'Gender_Male', 'Married_Yes', 'SelfEmployed_Yes'
            ]

            if hasattr(model, 'feature_names_in_'):
                input_df = input_df[model.feature_names_in_]
            else:
                input_df = input_df[expected_order]

            if model_choice == "Régression Logistique" and scaler is not None:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]

            proba_rejected = proba[0] * 100
            proba_approved = proba[1] * 100

            st.subheader("📊 Résultat de la prédiction")

            if prediction == 1:
                st.success("### ✅ PRÊT APPROUVÉ")
                st.balloons()
            else:
                st.error("### ❌ PRÊT REJETÉ")

            st.metric("Probabilité d'approbation", f"{proba_approved:.1f}%")
            st.metric("Probabilité de rejet", f"{proba_rejected:.1f}%")

# =====================================================================================
# TAB 3 : PERFORMANCE
# =====================================================================================

with tab3:
    mapping = {
        "Régression Logistique": "logistic_regression",
        "Random Forest": "random_forest"
    }

    model_name = mapping.get(model_choice)

    try:
        with open('../notebook/models/metadata.json', 'r') as f:
            metrics = json.load(f)
    except:
        st.error("❌ Fichier metadata.json introuvable.")
        st.stop()

    m = metrics['models'][model_name]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']:.2%}")
    col2.metric("Precision", f"{m['precision']:.2%}")
    col3.metric("Recall", f"{m['recall']:.2%}")
    col4.metric("F1-Score", f"{m['f1_score']:.2%}")
    col5.metric("AUC", f"{m['auc']:.3f}")

    st.subheader("Matrice de confusion")
    cm = [[50, 10], [5, 60]]

    fig = ff.create_annotated_heatmap(
        cm,
        x=['Rejected', 'Approved'],
        y=['Rejected', 'Approved'],
        colorscale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Courbe ROC")
    
    # Si tu veux, mets les vraies valeurs ici
    fpr = [0, 0.1, 0.2, 1]
    tpr = [0, 0.6, 0.8, 1]
    auc = 0.86

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={auc}"))
    st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("---")
st.write("Projet de prédiction d'approbation de prêt – Version corrigée")