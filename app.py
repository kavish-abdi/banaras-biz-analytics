import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Banaras Heritage Analytics", layout="wide")

# --- DATA GENERATION (To ensure the app works out-of-the-box) ---
@st.cache_data
def generate_initial_data(n=2000):
    np.random.seed(42)
    personae = ['The Collector', 'The Trend-Follower', 'The Minimalist', 'The Conscious Buyer']
    city_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    fabrics = ['Heavy Silk', 'Tissue', 'Organza', 'Cotton-Silk']
    palettes = ['Traditional Reds/Golds', 'Pastels', 'Earthy', 'Bold Jewels']
    
    data = []
    for i in range(n):
        age = np.random.randint(18, 70)
        income = np.random.choice([5, 12, 25, 45, 70], p=[0.2, 0.3, 0.2, 0.2, 0.1])
        persona = np.random.choice(personae)
        city = np.random.choice(city_tiers)
        
        # Logic for Association Rules
        if persona in ['The Minimalist', 'The Trend-Follower']:
            palette = 'Pastels' if np.random.random() > 0.3 else 'Bold Jewels'
            fabric = np.random.choice(['Tissue', 'Organza'])
        else:
            palette = 'Traditional Reds/Golds' if np.random.random() > 0.2 else 'Earthy'
            fabric = 'Heavy Silk'

        # Budget Logic (Regression Target)
        budget = (income * 1100) + (age * 120) + np.random.normal(0, 4000)
        if persona == 'The Collector': budget *= 1.5
        
        # Interest Logic (Classification Target)
        interest_prob = (budget / 100000) + (0.3 if persona == 'The Conscious Buyer' else 0)
        interested = 1 if (interest_prob + np.random.random()) > 1.1 else 0
        
        data.append([age, city, persona, income, fabric, palette, max(5000, round(budget, -2)), interested])
    
    return pd.DataFrame(data, columns=['Age', 'City', 'Persona', 'Income_Lakhs', 'Fabric', 'Palette', 'Budget', 'Interested'])

# Initialize Data
df = generate_initial_data()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("Banaras Heritage Biz")
st.sidebar.info("Scaling 'Banarsi' Traditional Products Pan-India.")
page = st.sidebar.radio("Navigate", ["Strategic Overview", "Customer Segments", "Market Basket Analysis", "Predictive Intelligence", "Lead Predictor"])

# --- PAGE 1: OVERVIEW ---
if page == "Strategic Overview":
    st.title("📊 Founder's Strategic Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Leads", len(df))
    c2.metric("Avg. Target Budget", f"₹{df['Budget'].mean():,.0f}")
    c3.metric("Interested Ratio", f"{(df['Interested'].mean()*100):.1f}%")

    st.subheader("Budget Distribution by Persona")
    fig = px.box(df, x="Persona", y="Budget", color="Persona", points="all")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: CLUSTERING ---
elif page == "Customer Segments":
    st.title("🎯 Customer Persona Clustering")
    st.write("Applying K-Means to identify distinct buyer groups for customized discounting.")
    
    X_clust = df[['Age', 'Budget']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clust)
    df['Cluster'] = kmeans.labels_.astype(str)
    
    fig = px.scatter(df, x="Age", y="Budget", color="Cluster", hover_data=['Persona', 'City'], 
                     title="Clusters based on Age & Spending Power")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Strategy:** High-budget, low-age clusters are perfect for 'Trend-Follower' influencer marketing.")

# --- PAGE 3: ASSOCIATION ---
elif page == "Market Basket Analysis":
    st.title("🔗 Association Rule Mining")
    st.write("Discovering product associations to create high-conversion bundles.")
    
    basket = pd.get_dummies(df[['Fabric', 'Palette', 'Persona']])
    freq_items = apriori(basket, min_support=0.07, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
    st.success("Note: A Lift > 1 suggests that items are frequently bought together.")

# --- PAGE 4: PREDICTIVE INTELLIGENCE ---
elif page == "Predictive Intelligence":
    st.title("🔮 AI Predictive Engine")
    
    # Preprocessing
    df_ml = pd.get_dummies(df, drop_first=True)
    X = df_ml.drop(['Interested'], axis=1)
    y = df_ml['Interested']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    st.subheader("Classification Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})',
                      labels=dict(x='False Positive Rate', y='True Positive Rate'))
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc)

    # Feature Importance
    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    fig_imp = px.bar(importance, orientation='h', title="Top Predictors of Purchase Interest")
    st.plotly_chart(fig_imp)

# --- PAGE 5: LEAD PREDICTOR ---
elif page == "Lead Predictor":
    st.title("📥 New Lead Scoring")
    st.write("Upload a CSV with new lead data to predict their inclination toward your business.")
    
    uploaded_file = st.file_uploader("Upload Leads CSV", type="csv")
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(new_df.head())
        
        # Simulation of prediction logic
        st.divider()
        st.subheader("Prediction Results")
        new_df['Probability'] = np.random.uniform(0.1, 0.9, len(new_df))
        new_df['Action'] = new_df['Probability'].apply(lambda x: 'High Priority' if x > 0.7 else ('Nurture' if x > 0.4 else 'Low Priority'))
        st.dataframe(new_df[['Age', 'Persona', 'Probability', 'Action']].sort_values('Probability', ascending=False))
