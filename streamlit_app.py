"""
Titanic Survival Predictor — streamlit_app.py
Maestría en Inteligencia Artificial y Aprendizaje Automático — UFHEC
Autora: Dominga Rodríguez
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ─── Configuración de la Página ─────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic AI — Dashboard de Inferencia",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Personalizado (Estética Premium Oscura) ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #05070f 0%, #0d1329 50%, #05070f 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(13, 19, 41, 0.95);
        border-right: 1px solid rgba(6, 182, 212, 0.1);
    }
    /* Tarjetas de métricas */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #06b6d4 !important;
        font-size: 28px !important;
        font-weight: 900 !important;
    }
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    .prediction-card {
        background: rgba(13, 19, 41, 0.7);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        backdrop-filter: blur(20px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 100px;
        padding: 10px 28px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(6, 182, 212, 0.4);
    }
    .info-box {
        background: rgba(6, 182, 212, 0.06);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ─── CARGA Y PROCESAMIENTO DE DATOS ──────────────────────────────────────────
@st.cache_data
def load_data():
    """Carga y procesa el dataset del Titanic."""
    df = pd.read_csv('titanic.csv')
    
    # Imputación
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Ingeniería de Características
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Codificaciones para modelar
    df_clean = df.copy()
    df_clean['Sex'] = df_clean['Sex'].map({'female': 0, 'male': 1})
    df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)
    df_clean['Embarked_Q'] = df_clean['Embarked_Q'].astype(int)
    df_clean['Embarked_S'] = df_clean['Embarked_S'].astype(int)
    
    return df, df_clean

@st.cache_resource
def train_models(df_clean):
    """Entrena los 4 modelos supervisados."""
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'Embarked_Q', 'Embarked_S']
    X = df_clean[feature_cols]
    y = df_clean['Survived']
    
    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar clasificadores
    lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(X_train, y_train)
    
    # Generar métricas de prueba
    metrics = {}
    for name, clf in [("Regresión Logística", lr), ("Árbol de Decisión", dt), ("KNN", knn), ("Random Forest", rf)]:
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds),
            'AUC-ROC': roc_auc_score(y_test, probs),
            'fpr': roc_curve(y_test, probs)[0].tolist(),
            'tpr': roc_curve(y_test, probs)[1].tolist(),
        }
        
    return scaler, lr, dt, knn, rf, metrics, feature_cols, X_test, y_test

# Cargar y entrenar
df_raw, df_clean = load_data()
scaler, lr, dt, knn, rf, metrics, feature_cols, X_test, y_test = train_models(df_clean)

# ─── SIDEBAR (NAVEGACIÓN) ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size: 40px; margin-bottom: 8px;'>🚢</div>
        <div style='font-size: 22px; font-weight: 900; color: #f1f5f9;'>TITANIC AI</div>
        <div style='font-size: 11px; color: #06b6d4; letter-spacing: 2px; text-transform: uppercase;'>Inferencia de Supervivencia</div>
    </div>
    <hr style='margin: 8px 0 20px; border-color: rgba(255,255,255,0.05);'>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navegar a:",
        ["🏠 Inicio — Proyecto", "📊 Explorador del Dataset", "📈 EDA Visual",
         "🔮 Predictor de Supervivencia", "🏆 Comparación de Modelos"]
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px; color:#94a3b8;'>
    <b style='color:#f1f5f9;'>📋 Dataset</b><br>
    · {len(df_raw)} registros de pasajeros<br>
    · {len(feature_cols)} variables predictoras<br>
    · Clasificación Binaria<br><br>
    <b style='color:#f1f5f9;'>🎓 Maestría IA</b><br>
    · Dominga Rodríguez · UFHEC<br>
    · Aprendizaje Autónomo - 2026
    </div>
    """, unsafe_allow_html=True)

# ─── PÁGINA: INICIO ──────────────────────────────────────────────────────────
if page == "🏠 Inicio — Proyecto":
    st.markdown("""
    <div class='hero-card'>
        <h1 style='font-size:42px; font-weight:900; margin:0; background:linear-gradient(135deg,#06b6d4,#8b5cf6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Predicción de Supervivencia del Titanic
        </h1>
        <p style='color:#94a3b8; font-size:18px; margin-top:12px; max-width:800px;'>
            Aplicación de aprendizaje supervisado de clasificación para predecir si un pasajero sobrevivió basándose en variables demográficas y de viaje como edad, clase, tarifa y familia.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pasajeros", f"{len(df_raw):,}")
    with col2:
        st.metric("Tasa de Supervivencia", f"{(df_raw['Survived'].mean()*100):.1f}%")
    with col3:
        st.metric("Accuracy Random Forest", f"{(metrics['Random Forest']['Accuracy']*100):.1f}%")
    with col4:
        st.metric("F1-Score RF", f"{metrics['Random Forest']['F1-Score']:.3f}")
        
    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        ### ⚠️ El Desastre Histórico
        El 15 de abril de 1912, el RMS Titanic chocó contra un iceberg y se hundió, provocando la muerte de más de 1,500 personas de las 2,224 a bordo. Los botes salvavidas eran insuficientes y la evacuación estuvo marcada por normas sociales del período y la distribución física de las clases de pasaje.
        
        ### 🎯 Objetivos del Proyecto
        *   **Clasificar:** Determinar binariamente si un perfil de pasajero sobrevive (1) o fallece (0).
        *   **Explicar:** Extraer la importancia de variables que determinaron la vida o muerte de los pasajeros.
        *   **Comparar:** Evaluar cuantitativamente 4 modelos supervisados clásicos.
        """)
    with col_r:
        st.markdown("""
        ### 🔧 Características del Dashboard
        *   **Explorador interactivo:** Visualiza y filtra la base de datos limpia.
        *   **EDA visual:** Gráficos estéticos e intuitivos de supervivencia agrupada.
        *   **Predictor instantáneo:** Simula perfiles ajustando barras deslizantes y botones.
        *   **Evaluación del modelo:** Compara curvas ROC y métricas de error.
        """)

# ─── PÁGINA: EXPLORADOR DEL DATASET ──────────────────────────────────────────
elif page == "📊 Explorador del Dataset":
    st.markdown("## 📊 Explorador del Dataset")
    st.markdown("*Manifiesto de pasajeros del Titanic procesado mediante el pipeline ETL*")
    
    st.markdown("### Vista Previa de los Datos (Primeros 15 registros)")
    st.dataframe(df_raw.head(15), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Estadísticas Descriptivas Generales")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)
    with col2:
        st.markdown("### Valores Faltantes en el Dataset Original")
        nulos = pd.read_csv('titanic.csv').isnull().sum()
        st.dataframe(pd.DataFrame({"Nulos": nulos}), use_container_width=True)

# ─── PÁGINA: EDA VISUAL ──────────────────────────────────────────────────────
elif page == "📈 EDA Visual":
    st.markdown("## 📈 Análisis Exploratorio de Datos (EDA)")
    
    # Heatmap
    st.markdown("### Mapa de Calor de Correlaciones")
    corr = df_clean.select_dtypes(include=[np.number]).corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Matriz de Correlación Lineal"
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9')
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Tasa de Supervivencia según Género")
        gender_surv = df_raw.groupby('Sex')['Survived'].mean().reset_index()
        gender_surv['Survived'] *= 100
        fig_gender = px.bar(
            gender_surv, x='Sex', y='Survived',
            color='Sex', color_discrete_map={'female': '#06b6d4', 'male': '#8b5cf6'},
            labels={'Sex': 'Género', 'Survived': 'Porcentaje de Supervivencia (%)'},
            title='Supervivencia por Sexo (Protocolo \"Mujeres y Niños primero\")'
        )
        fig_gender.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'))
        st.plotly_chart(fig_gender, use_container_width=True)
        
    with col2:
        st.markdown("### Tasa de Supervivencia según Clase del Boleto")
        class_surv = df_raw.groupby('Pclass')['Survived'].mean().reset_index()
        class_surv['Survived'] *= 100
        fig_class = px.bar(
            class_surv, x='Pclass', y='Survived',
            color='Pclass', color_continuous_scale=px.colors.sequential.Viridis,
            labels={'Pclass': 'Clase', 'Survived': 'Porcentaje de Supervivencia (%)'},
            title='Supervivencia por Estatus Socioeconómico (Pclass)'
        )
        fig_class.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'))
        st.plotly_chart(fig_class, use_container_width=True)

# ─── PÁGINA: PREDICTOR EN TIEMPO REAL ───────────────────────────────────────
elif page == "🔮 Predictor de Supervivencia":
    st.markdown("## 🔮 Predictor de Supervivencia en Tiempo Real")
    st.markdown("*Ajusta las características del pasajero y selecciona el clasificador*")
    
    col_inputs, col_results = st.columns([1.2, 1])
    
    with col_inputs:
        st.markdown("### ⚙️ Características del Pasajero")
        with st.form("predictor_form"):
            c1, c2 = st.columns(2)
            with c1:
                sex = st.radio("Género (Sex):", ["Femenino (female)", "Masculino (male)"])
                pclass = st.radio("Clase del Boleto (Pclass):", [1, 2, 3], index=2)
                age = st.slider("Edad en años (Age):", 0, 80, 25)
                fare = st.slider("Tarifa Pagada (Fare):", 0.0, 512.0, 15.0, 0.5)
            with c2:
                sibsp = st.slider("Hermanos/Cónyuges a bordo (SibSp):", 0, 8, 0)
                parch = st.slider("Padres/Hijos a bordo (Parch):", 0, 6, 0)
                embarked = st.radio("Puerto de Embarque (Embarked):", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)
                
            model_choice = st.selectbox(
                "Algoritmo de Inferencia:",
                ["Random Forest (Mejor Desempeño)", "Regresión Logística (Interpretable)", "Árbol de Decisión", "K-Nearest Neighbors"]
            )
            submit = st.form_submit_button("🔮 Estimar Supervivencia")
            
    with col_results:
        if submit:
            # Preprocesar
            sex_num = 0 if "Femenino" in sex else 1
            pclass_num = int(pclass)
            emb_c = 1 if "Cherbourg" in embarked else 0
            emb_q = 1 if "Queenstown" in embarked else 0
            emb_s = 1 if "Southampton" in embarked else 0
            
            family_size = sibsp + parch + 1
            is_alone = 1 if family_size == 1 else 0
            
            # Vector raw
            raw_data = np.array([[pclass_num, sex_num, age, sibsp, parch, fare, is_alone, emb_q, emb_s]])
            # Escalar
            scaled_data = scaler.transform(raw_data)
            
            # Clasificar según modelo elegido
            if "Random Forest" in model_choice:
                prob = rf.predict_proba(scaled_data)[0, 1]
                model_name = "Random Forest"
            elif "Regresión Logística" in model_choice:
                prob = lr.predict_proba(scaled_data)[0, 1]
                model_name = "Regresión Logística"
            elif "Árbol de Decisión" in model_choice:
                prob = dt.predict_proba(scaled_data)[0, 1]
                model_name = "Árbol de Decisión"
            else:
                prob = knn.predict_proba(scaled_data)[0, 1]
                model_name = "K-Nearest Neighbors"
                
            survives = prob > 0.5
            color = "#10b981" if survives else "#ef4444"
            outcome = "SOBREVIVE" if survives else "NO SOBREVIVE"
            
            # Card de resultado
            st.markdown(f"""
            <div class='prediction-card'>
                <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;'>
                    Probabilidad de Supervivencia
                </div>
                <div style='font-size:72px; font-weight:900; color:{color}; line-height:1;'>
                    {(prob*100):.1f}%
                </div>
                <hr style='border-color:rgba(255,255,255,0.07); margin:16px 0;'>
                <div style='font-size:24px; font-weight:900; color:{color}; margin-bottom:8px;'>{outcome}</div>
                <hr style='border-color:rgba(255,255,255,0.07); margin:16px 0;'>
                <div style='font-size:12px; color:#94a3b8;'>
                    Modelo utilizado: <span style='color:#06b6d4;'>{model_name}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                    'bar': {'color': color},
                    'bgcolor': 'rgba(255, 255, 255, 0.03)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(255, 255, 255, 0.05)',
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                height=220,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.markdown("""
            <div class='prediction-card'>
                <div style='font-size:48px; margin-bottom:12px;'>🔮</div>
                <div style='font-size:15px; color:#94a3b8;'>
                    Ingresa los datos en el formulario y presiona <strong>Estimar Supervivencia</strong> para ejecutar el modelo.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── PÁGINA: COMPARACIÓN DE MODELOS ──────────────────────────────────────────
elif page == "🏆 Comparación de Modelos":
    st.markdown("## 🏆 Comparación de Modelos de Machine Learning")
    st.markdown("*Evaluación cuantitativa en el conjunto de prueba (30% de validación)*")
    
    # Crear dataframe comparativo de métricas
    comparison_data = []
    for name, data in metrics.items():
        comparison_data.append({
            'Modelo': name,
            'Accuracy': f"{data['Accuracy']*100:.2f}%",
            'Precision': f"{data['Precision']*100:.2f}%",
            'Recall': f"{data['Recall']*100:.2f}%",
            'F1-Score': f"{data['F1-Score']:.4f}",
            'AUC-ROC': f"{data['AUC-ROC']:.4f}"
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Graficar curvas ROC
    st.markdown("### Curvas ROC Comparativas (Conjunto de Prueba)")
    fig_roc = go.Figure()
    colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']
    for i, (name, data) in enumerate(metrics.items()):
        fig_roc.add_trace(go.Scatter(
            x=data['fpr'], y=data['tpr'],
            mode='lines',
            name=f"{name} (AUC = {data['AUC-ROC']:.3f})",
            line=dict(color=colors[i], width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Clasificación Aleatoria',
        line=dict(color='gray', dash='dash')
    ))
    fig_roc.update_layout(
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        height=450
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importances of Random Forest
    st.markdown("### Importancia de Características (Random Forest)")
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': [f.replace('_Score', '').replace('_', ' ') for f in feature_cols],
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(
        feat_imp_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#8b5cf6', '#06b6d4'],
        title='Importancia Relativa de Variables Predictoras'
    )
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        height=380,
        showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)
