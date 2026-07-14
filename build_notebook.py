import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load data to verify and get coefficients
df = pd.read_csv('titanic.csv')

# Handle missing values
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna('S')
df['HasCabin'] = df['Cabin'].notna().astype(int)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Map Sex: female=0, male=1
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# Embarked dummy variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# Ensure columns are 0 or 1 (integer)
df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

# Features and target
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'Embarked_Q', 'Embarked_S']
X = df[feature_cols]
y = df['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)

# Fit Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)

# Fit KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Fit Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get metrics
models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Random Forest": rf
}

print("=== Model Metrics ===")
for name, clf in models.items():
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

print("\n=== Logistic Regression Parameters ===")
print("Feature columns:", feature_cols)
print("Coefficients:", lr.coef_[0].tolist())
print("Intercept:", lr.intercept_[0])
print("Scaler mean:", scaler.mean_.tolist())
print("Scaler scale (std dev):", scaler.scale_.tolist())

# Generate notebook.ipynb JSON structure
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Proyecto Final de Aprendizaje Autónomo: Predicción de Supervivencia del Titanic\n",
    "**Estudiante:** Dominga Rodríguez  \n",
    "**Maestría en Inteligencia Artificial y Aprendizaje Automático — UFHEC**  \n\n",
    "Este notebook contiene el análisis de Machine Learning completo para predecir si un pasajero sobrevivió al naufragio del Titanic utilizando variables demográficas y de viaje. Se sigue la metodología CRISP-ML(Q) estructurada de la siguiente manera:\n",
    "1. **Fase ETL (Extract, Transform, Load)**: Carga de datos, imputación inteligente de nulos y creación de variables.\n",
    "2. **Fase EDA (Exploratory Data Analysis)**: Exploración visual de patrones de supervivencia.\n",
    "3. **Preprocesamiento**: Estandarización y división del conjunto de datos.\n",
    "4. **Modelado**: Entrenamiento de 4 algoritmos supervisados (Regresión Logística, Árbol de Decisión, KNN, Random Forest).\n",
    "5. **Evaluación de Modelos**: Comparación cuantitativa de métricas e interpretación de coeficientes.\n",
    "6. **Simulador Interactivo**: Widget gráfico para probar predicciones en tiempo real."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importación de librerías esenciales\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, confusion_matrix, roc_curve, classification_report\n",
    ")\n\n",
    "# Configuración visual de los gráficos\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
    "plt.rcParams['figure.figsize'] = (10, 6)\n",
    "plt.rcParams['font.size'] = 11"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Fase ETL (Extract, Transform, Load)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extracción: Cargar el dataset\n",
    "df = pd.read_csv('titanic.csv')\n",
    "print(f\"Dimensiones originales: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verificación de valores nulos\n",
    "print(\"Valores nulos por columna:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transformación: Imputación inteligente\n",
    "# Imputar la edad según la mediana agrupada por Pclass y Sex\n",
    "df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))\n\n",
    "# Imputar Embarked con la moda ('S')\n",
    "df['Embarked'] = df['Embarked'].fillna('S')\n\n",
    "# Creación de la característica binaria de cabina\n",
    "df['HasCabin'] = df['Cabin'].notna().astype(int)\n\n",
    "# Ingeniería de Características: Tamaño de la familia y si viaja solo\n",
    "df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
    "df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n\n",
    "# Codificación numérica de Sex (female=0, male=1)\n",
    "df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})\n\n",
    "# Codificación One-Hot para Embarked\n",
    "df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)\n",
    "df['Embarked_Q'] = df['Embarked_Q'].astype(int)\n",
    "df['Embarked_S'] = df['Embarked_S'].astype(int)\n\n",
    "# Eliminar columnas no requeridas para el modelado directo\n",
    "df_clean = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')\n\n",
    "print(f\"Dimensiones después del ETL: {df_clean.shape}\")\n",
    "df_clean.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Fase EDA (Exploratory Data Analysis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribución de la supervivencia por género\n",
    "sns.barplot(x='Sex', y='Survived', data=df, hue='Sex', legend=False)\n",
    "plt.title('Tasa de Supervivencia según Género (0 = Femenino, 1 = Masculino)')\n",
    "plt.ylabel('Proporción de Supervivencia')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribución de supervivencia por Clase de Pasajero\n",
    "sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')\n",
    "plt.title('Tasa de Supervivencia según la Clase del Boleto (Pclass)')\n",
    "plt.ylabel('Proporción de Supervivencia')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mapa de calor de correlaciones del conjunto limpio\n",
    "sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
    "plt.title('Mapa de Calor de Correlaciones Lineales')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Preprocesamiento de Datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Definir características y variable objetivo\n",
    "features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'IsAlone', 'Embarked_Q', 'Embarked_S']\n",
    "X = df_clean[features]\n",
    "y = df_clean['Survived']\n\n",
    "# División de datos en Entrenamiento (70%) y Prueba (30%) estratificado\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.3, random_state=42, stratify=y\n",
    ")\n\n",
    "# Estandarización de las características para asegurar convergencia en modelos métricos y lineales\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n\n",
    "print(f\"Entrenamiento: {X_train_scaled.shape}, Prueba: {X_test_scaled.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Modelado y Entrenamiento"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inicialización de los 4 clasificadores supervisados\n",
    "models = {\n",
    "    'Regresión Logística': LogisticRegression(random_state=42),\n",
    "    'Árbol de Decisión': DecisionTreeClassifier(max_depth=5, random_state=42),\n",
    "    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),\n",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)\n",
    "}\n\n",
    "# Entrenamiento de cada modelo\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train_scaled, y_train)\n",
    "    print(f\"Modelo '{name}' entrenado con éxito.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluación Cualitativa y Métricas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tabla de comparación de métricas en el conjunto de prueba\n",
    "results = []\n",
    "plt.figure(figsize=(10, 8))\n\n",
    "for name, model in models.items():\n",
    "    preds = model.predict(X_test_scaled)\n",
    "    probs = model.predict_proba(X_test_scaled)[:, 1]\n\n",
    "    # Métricas clave\n",
    "    acc = accuracy_score(y_test, preds)\n",
    "    prec = precision_score(y_test, preds)\n",
    "    rec = recall_score(y_test, preds)\n",
    "    f1 = f1_score(y_test, preds)\n",
    "    auc = roc_auc_score(y_test, probs)\n\n",
    "    results.append({\n",
    "        'Modelo': name,\n",
    "        'Accuracy': acc,\n",
    "        'Precision': prec,\n",
    "        'Recall': rec,\n",
    "        'F1-Score': f1,\n",
    "        'AUC-ROC': auc\n",
    "    })\n\n",
    "    # Curvas ROC\n",
    "    fpr, tpr, _ = roc_curve(y_test, probs)\n",
    "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {auc:.3f})\")\n\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Clasificación Aleatoria')\n",
    "plt.xlabel('Tasa de Falsos Positivos (FPR)')\n",
    "plt.ylabel('Tasa de Verdaderos Positivos (TPR)')\n",
    "plt.title('Curvas ROC Comparativas de Modelos')\n",
    "plt.legend(loc='lower right')\n",
    "plt.show()\n\n",
    "df_results = pd.DataFrame(results)\n",
    "df_results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Coeficientes de la Regresión Logística para el despliegue\n",
    "lr_model = models['Regresión Logística']\n",
    "print(\"Intercepto:\", lr_model.intercept_[0])\n",
    "print(\"Coeficientes por característica:\")\n",
    "for feat, coef in zip(features, lr_model.coef_[0]):\n",
    "    print(f\"  {feat}: {coef:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Simulador Predictivo Interactivo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Widget simple de simulación en Python\n",
    "from ipywidgets import interact, widgets\n\n",
    "def simular_supervivencia(pclass, sex, age, sibsp, parch, fare, is_alone, emb_q, emb_s):\n",
    "    # Construir registro\n",
    "    record = np.array([[pclass, sex, age, sibsp, parch, fare, is_alone, emb_q, emb_s]])\n",
    "    # Escalar\n",
    "    record_scaled = scaler.transform(record)\n",
    "    \n",
    "    # Clasificar con Random Forest\n",
    "    prob = models['Random Forest'].predict_proba(record_scaled)[0, 1]\n",
    "    surv = \"SOBREVIVE\" if prob > 0.5 else \"NO SOBREVIVE\"\n",
    "    \n",
    "    print(f\"Resultado estimado: {surv}\")\n",
    "    print(f\"Probabilidad de supervivencia: {prob*100:.2f}%\")\n\n",
    "interact(\n",
    "    simular_supervivencia,\n",
    "    pclass=[1, 2, 3],\n",
    "    sex={'Femenino': 0, 'Masculino': 1},\n",
    "    age=widgets.IntSlider(min=0, max=80, step=1, value=25),\n",
    "    sibsp=widgets.IntSlider(min=0, max=8, step=1, value=0),\n",
    "    parch=widgets.IntSlider(min=0, max=6, step=1, value=0),\n",
    "    fare=widgets.FloatSlider(min=0.0, max=512.0, step=1.0, value=32.0),\n",
    "    is_alone=[( 'Sí', 1), ('No', 0)],\n",
    "    emb_q=[('Sí', 1), ('No', 0)],\n",
    "    emb_s=[('Sí', 1), ('No', 0)]\n",
    ")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipywidgets",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat_minor": 2
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\nNotebook guardado exitosamente en: notebook.ipynb")
