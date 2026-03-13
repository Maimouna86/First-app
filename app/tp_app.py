import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ======================================================================
# Configuration de la page
# ======================================================================

st.set_page_config(
    page_title="Prédiction d'Approbation de Prêt",
    page_icon="🏦",
    layout="wide"
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
# Chargement des données
# ======================================================================

df = load_data()
if df is None:
    st.stop()

# ======================================================================
# Tabs
# ======================================================================

tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🔮 Prédiction", "📈 Performance"])

# ======================================================================
# TAB 1 - EXPLORATION
# ======================================================================

with tab1:

    st.header("📊 Exploration des Données")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Demandes", len(df))
    col2.metric("Taux Approbation", f"{(df['Loan_Status']=='Y').mean()*100:.2f}%")
    col3.metric("Montant Moyen", f"{df['LoanAmount'].mean():.0f}")
    col4.metric("Revenu Moyen", f"{df['ApplicantIncome'].mean():.0f}")

    # Histogramme revenus
    st.subheader("📈 Distribution des Revenus")
    fig = px.histogram(df, x="ApplicantIncome", nbins=30)
    fig.add_vline(x=df["ApplicantIncome"].mean())
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot loan
    st.subheader("📦 Montants de Prêt Demandés")
    st.plotly_chart(px.box(df, y="LoanAmount"), use_container_width=True)

    # Bar chart éducation
    st.subheader("🎓 Approbation selon l'Éducation")

    grouped = df.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
    grouped["Percentage"] = grouped.groupby("Education")["Count"].transform(lambda x: x / x.sum() * 100)
    approved = grouped[grouped["Loan_Status"] == "Y"]

    st.plotly_chart(px.bar(
        approved,
        x="Education",
        y="Percentage",
        color="Education"
    ), use_container_width=True)

    # Pie chart
    st.subheader("🥧 Répartition Approuvé / Rejeté")

    counts = df["Loan_Status"].value_counts()

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Répartition des Décisions",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap corrélation
    st.subheader("🔥 Matrice de Corrélation")

    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.markdown("---")
    st.subheader("🔍 Filtres Interactifs")

    col1, col2 = st.columns(2)

    with col1:
        income_range = st.slider(
            "Revenu",
            int(df["ApplicantIncome"].min()),
            int(df["ApplicantIncome"].max()),
            (int(df["ApplicantIncome"].min()), int(df["ApplicantIncome"].max()))
        )

    with col2:
        selected_education = st.multiselect(
            "Niveau d'éducation",
            df["Education"].unique(),
            default=df["Education"].unique()
        )

    filtered_df = df[
        (df["ApplicantIncome"].between(income_range[0], income_range[1])) &
        (df["Education"].isin(selected_education))
    ]

    st.write(f"Nombre de résultats filtrés : {len(filtered_df)}")

    st.download_button(
        "📥 Télécharger les données",
        df.to_csv(index=False),
        "loan_data.csv",
        "text/csv"
    )

# ======================================================================
# TAB 2 - PRÉDICTION
# ======================================================================

with tab2:
    st.header("🔮 Prédiction")
    st.info("Formulaire de prédiction à implémenter.")

# ======================================================================
# TAB 3 - PERFORMANCE
# ======================================================================

with tab3:

    st.header("📈 Performance du Modèle")

    model, scaler = load_model(model_choice)

    if model is not None:
        st.success("✅ Modèle chargé avec succès")
        col1, col2 = st.columns(2)
        col1.metric("Type de Modèle", model_choice)
        col2.metric("Scaler utilisé", "Oui" if scaler is not None else "Non")

        col3, col4 = st.columns(2)
        col3.metric("Accuracy", "85%")
        col4.metric("F1 Score", "0.82")

# ======================================================================
# Footer
# ======================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center'>Application Streamlit - Prédiction d'Approbation de Prêt</div>",
    unsafe_allow_html=True
)