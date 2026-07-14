# Titanic AI — Predictor de Supervivencia con Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/domingamaestriaufhec/titanic-ml-predictor/blob/main/notebook.ipynb)
[![Simulador Web](https://img.shields.io/badge/Simulador%20Web-Abrir-cyan?style=flat-square&logo=html5)](https://domingamaestriaufhec.github.io/titanic-ml-predictor/app/index.html)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-ml-predictor.streamlit.app)

Este repositorio contiene un proyecto completo de Machine Learning supervisado para predecir la supervivencia de pasajeros a bordo del RMS Titanic basándose en variables demográficas y de viaje.

---

## 📁 Estructura del Repositorio

```
├── titanic.csv                 # Dataset original
├── notebook.ipynb              # Notebook de entrenamiento (ETL, EDA, 4 Modelos, Métricas)
├── streamlit_app.py            # Dashboard analítico de Streamlit
├── generate_titanic_docx.py    # Compilador del reporte Word APA
├── Documento_Titanic.docx      # Reporte académico formal
├── Documento_Titanic.md        # Reporte de investigación en Markdown
├── requirements.txt            # Dependencias del sistema
├── index.html                  # Landing Page explicativa (Ciclo de Vida CRISP-ML(Q))
├── ar_titanic_simulation.png   # Imagen de Realidad Aumentada generada por IA
└── app/
    ├── index.html              # Simulador interactivo local (HTML5)
    ├── style.css               # Estilos premium del simulador (Glassmorphism)
    └── script.js               # Motor de inferencia en JavaScript cliente
```

---

## 🛠️ Requisitos de Instalación (Python)

Para ejecutar el cuaderno de Jupyter y el dashboard de Streamlit, se requiere Python 3.9 o superior. Sigue estos pasos para preparar el entorno:

1. **Clonar o descargar el repositorio:**
   ```bash
   git clone https://github.com/domingamaestriaufhec/titanic-ml-predictor.git
   cd "titanic datos"
   ```

2. **Crear e inicializar un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   # En Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Guías de Ejecución

### 1. Cuaderno de Jupyter (`notebook.ipynb`)
Para abrir el cuaderno interactivo de modelado, ejecuta:
```bash
jupyter notebook notebook.ipynb
```
*El cuaderno contiene el proceso metodológico completo bajo el ciclo de vida CRISP-ML(Q), entrenando y comparando Regresión Logística, Árbol de Decisión, KNN y Random Forest.*

### 2. Dashboard de Streamlit (`streamlit_app.py`)
Para lanzar el dashboard predictivo en tu navegador local, ejecuta:
```bash
streamlit run streamlit_app.py
```

### 3. Simulador Web Local (`app/index.html`)
No requiere ningún servidor. Simplemente haz doble clic sobre el archivo [index.html](file:///c:/Users/HP/Downloads/titanic%20datos/app/index.html) para abrir el simulador en cualquier navegador moderno (Edge, Chrome, Firefox, Safari).

### 4. Compilación del Reporte de Word
Si modificas el reporte de texto y deseas volver a compilar el documento APA formal, ejecuta:
```bash
python generate_titanic_docx.py
```

---

## 🎓 Metodología CRISP-ML(Q)
El proyecto ha sido desarrollado siguiendo las etapas estructuradas del ciclo de vida de proyectos de IA:
1. **Entendimiento del Negocio/Problema**
2. **Entendimiento de los Datos**
3. **Preparación de los Datos (ETL)**
4. **Modelado y Algoritmos**
5. **Evaluación de Métricas**
6. **Despliegue (Deployment)**

---

## ✍️ Autora
**Dominga Rodríguez**  
Universidad Francisco Henríquez y Carvajal (UFHEC)  
Maestría en Inteligencia Artificial y Aprendizaje Automático
