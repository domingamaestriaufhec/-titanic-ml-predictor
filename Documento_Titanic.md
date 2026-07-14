# Modelo Predictivo de Supervivencia en el Siniestro del Titanic mediante Aprendizaje Supervisado de Machine Learning

**Dominga Rodríguez**  
**Matrícula:** [Matrícula del Estudiante]  
**Licda. en Educación mención English**  
**Maestría en Inteligencia Artificial y Aprendizaje Automático**  
**Universidad Francisco Henríquez y Carvajal (UFHEC)**  
**Materia:** Aprendizaje Autónomo  
**Facilitador:** [Nombre del Facilitador]  
**Fecha:** 14 de julio de 2026

---

## Introducción

El hundimiento del RMS Titanic en abril de 1912 sigue siendo uno de los desastres marítimos más trágicos e influyentes en la historia moderna. De los 2,224 pasajeros y tripulantes a bordo, más de 1,500 perdieron la vida cuando el barco chocó contra un iceberg en el Atlántico Norte. Esta catástrofe no solo conmocionó al mundo, sino que también impulsó reformas fundamentales en las regulaciones de seguridad marítima internacional, incluyendo la obligatoriedad de botes salvavidas suficientes para todas las personas a bordo.

Desde una perspectiva analítica y de ciencia de datos, el desastre del Titanic presenta un escenario de estudio único. A pesar de que el naufragio estuvo rodeado de caos y pánico, la supervivencia de los pasajeros no fue completamente aleatoria. Factores estructurales, normas sociales de la época (como el protocolo de "mujeres y niños primero"), la ubicación de los camarotes según la clase socioeconómica y la estructura del tamaño de las familias desempeñaron un papel decisivo en la determinación de quién sobrevivió y quién pereció.

El presente proyecto plantea el diseño y desarrollo de un modelo predictivo basado en Machine Learning Supervisado para estimar la probabilidad de supervivencia de un pasajero en función de sus características socio-demográficas y variables de viaje. Mediante la comparación de múltiples algoritmos supervisados de clasificación, este sistema busca modelar con precisión los patrones de mortalidad del naufragio, sirviendo como una demostración práctica de cómo los métodos predictivos pueden extraer conocimiento valioso de bases de datos históricas estructuradas.

---

## Antecedentes

El análisis predictivo sobre los datos del Titanic se ha convertido en un punto de referencia (benchmark) fundamental en la literatura de la ciencia de datos y el aprendizaje automático. Diversas investigaciones han explorado cómo variables específicas determinan las probabilidades de supervivencia, validando empíricamente las crónicas históricas del naufragio.

Estudios clásicos de sociología y economía del comportamiento, como los de Frey et al. (2011), confirman que las normas sociales de caballerosidad ("mujeres y niños primero") y los incentivos de supervivencia interactuaron directamente con la clase socioeconómica del pasajero. Sus hallazgos demuestran que los pasajeros de primera clase tuvieron un acceso preferencial a los botes salvavidas, lo que se tradujo en una tasa de supervivencia significativamente superior en comparación con la tercera clase, donde el hacinamiento y la falta de información limitaron la evacuación.

Por otro lado, la aplicación de algoritmos modernos de clasificación sobre esta base de datos ha evidenciado que los modelos basados en árboles de decisión (como Random Forest y Gradient Boosting) superan a los modelos lineales al capturar interacciones complejas no lineales (por ejemplo, el efecto combinado de ser niño y viajar en tercera clase). Investigadores en computación educativa y analítica predictiva destacan que el dataset del Titanic permite ilustrar de manera transparente el impacto del preprocesamiento de datos (ETL) y el tratamiento de valores faltantes en la precisión final del modelo.

---

## Definición del Problema y Pregunta de Investigación

### Pregunta de Investigación Orientada a Machine Learning
¿Cómo optimizar la predicción de supervivencia de un pasajero del Titanic mediante un modelo de Machine Learning supervisado de clasificación binaria, utilizando variables demográficas y de viaje como el sexo, la edad, la clase de boleto, la tarifa y el tamaño de la familia?

### Alcance de Predicción del Modelo
El modelo propuesto permitirá:
*   **Predecir:** Si un pasajero sobrevivió (1) o no (0) al naufragio basándose en su perfil de datos.
*   **Clasificar:** A los pasajeros en grupos de alta o baja probabilidad de supervivencia para auditar las disparidades del evento.
*   **Identificar:** Qué variables (como la clase o el sexo) tuvieron el mayor peso o importancia predictiva en el desenlace del siniestro.

### ¿A quién impacta?
Este proyecto tiene un fuerte carácter didáctico y académico, impactando directamente a estudiantes y docentes de la Maestría en Inteligencia Artificial y Aprendizaje Automático en la Universidad Francisco Henríquez y Carvajal (UFHEC), al proporcionar un pipeline transparente y reproducible bajo la metodología de desarrollo de ciclo de vida CRISP-ML(Q).

---

## Arquitectura del Modelo de Machine Learning y Decisiones

### Variable Objetivo (Target)
*   **`Survived` (Clasificación Binaria):**
    *   **1** = Sobrevivió
    *   **0** = No sobrevivió (Falleció)

### Variables Independientes (Features)
*   **`Pclass` (Clase de Pasajero):** Variable ordinal (1.ª, 2.ª o 3.ª clase) que actúa como proxy del estatus socioeconómico.
*   **`Sex` (Género):** Variable categórica binaria (masculino o femenino).
*   **`Age` (Edad):** Variable continua representativa del grupo de edad (bebés, niños, adultos, ancianos).
*   **`Fare` (Tarifa de Pasaje):** Valor numérico del costo del boleto.
*   **`FamilySize` (Tamaño de Familia):** Característica de ingeniería calculada como la suma de hermanos/cónyuges (`SibSp`) más padres/hijos (`Parch`) más 1 (el propio pasajero).
*   **`IsAlone` (Si viaja solo):** Variable binaria indicando si el pasajero no tenía familiares a bordo.
*   **`Embarked` (Puerto de Embarque):** Categorías correspondientes a C (Cherbourg), Q (Queenstown) o S (Southampton).

### Enfoque de Machine Learning Evaluado
Para cumplir con los requisitos académicos, se evalúan y comparan **cuatro modelos de clasificación supervisada**:
1.  **Regresión Logística:** Proporciona un modelo base probabilístico e interpretable donde los coeficientes representan los odds-ratio de supervivencia.
2.  **Árbol de Decisión:** Genera reglas lógicas intuitivas estructuradas en forma de diagrama de flujo educativo.
3.  **Random Forest:** Ensamble de árboles de decisión que mejora la generalización y proporciona una estimación robusta de la importancia de las características.
4.  **K-Nearest Neighbors (KNN):** Clasifica en función de la distancia y similitud con otros perfiles de pasajeros en el espacio de características.

---

### Tabla 1: Arquitectura de Datos del Sistema y Decisiones Tecnológicas

| Componente | Descripción |
| :--- | :--- |
| **Tipo de IA** | Machine Learning Supervisado. El modelo aprende a partir de etiquetas reales de supervivencia (`Survived`). |
| **Problema** | Clasificación binaria (Clases: 1 / 0). |
| **Algoritmo Principal** | **Random Forest Classifier** (por su alto desempeño ante interacciones no lineales). |
| **Algoritmo Base e Interpretable** | **Regresión Logística** (para la extracción directa de coeficientes e integración en el frontend JS). |
| **Variables del Dataset** | Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone. |
| **Métricas de Evaluación** | Accuracy, Precision, Recall, F1-Score y Área bajo la curva ROC (AUC-ROC). |
| **Tecnologías de Desarrollo** | Python, Scikit-learn, Pandas, NumPy, Streamlit (Dashboard), HTML5/CSS3/JavaScript (App local). |

---

### Justificación del Modelo Seleccionado

Se seleccionan **Random Forest** y **Regresión Logística** como los pilares del proyecto por las siguientes razones:
1.  **Desempeño y Manejo de No Linealidades (Random Forest):** Los datos demuestran que las decisiones de evacuación no fueron lineales (por ejemplo, los hombres de primera clase sobrevivieron en menor proporción que las mujeres de tercera clase). Random Forest captura de forma nativa estas relaciones mediante la división aleatoria del espacio de características sin necesidad de ajustar transformaciones complejas de variables.
2.  **Explicabilidad y Despliegue Ligero (Regresión Logística):** La regresión logística ofrece una ecuación matemática cerrada. Sus coeficientes permiten interpretar directamente cómo el incremento en una unidad de una variable afecta las probabilidades relativas de supervivencia. Además, esta ecuación lineal simple se puede traducir fácilmente a código JavaScript puro en el cliente para lograr una respuesta inmediata en el navegador sin depender de un servidor de backend costoso.

---

## Datos Necesarios para el Desarrollo del Modelo

### Tabla 2: Tipos de Datos Requeridos y Fuentes de Obtención

| Tipo de dato | Descripción | Fuente de obtención | Tecnología / Formato |
| :--- | :--- | :--- | :--- |
| **Datos Demográficos** | Género, edad y estatus familiar del pasajero. | Dataset histórico del Titanic. | CSV (`titanic.csv`) |
| **Datos de Viaje** | Clase de cabina, puerto de embarque, tarifa y código de boleto. | Registros de la lista de pasajeros del RMS Titanic. | CSV / Pandas DataFrame |

---

## Datos y Analítica para la Toma de Decisiones

### Proceso ETL (Extract, Transform, Load)

#### Extract (Extracción)
Los datos brutos provienen del archivo [titanic.csv](file:///c:/Users/HP/Downloads/titanic%20datos/titanic.csv), el cual representa el manifiesto estructurado de los pasajeros a bordo del buque.

#### Transform (Transformación)
1.  **Imputación de Valores Faltantes:**
    *   La columna `Age` presenta alrededor del 20% de valores nulos. Estos se imputan utilizando la mediana agrupada por `Pclass` y `Sex` para mantener la consistencia demográfica.
    *   La columna `Embarked` tiene 2 valores nulos, los cuales se rellenan con la moda ('S').
    *   La columna `Cabin` presenta más del 70% de nulos; se transforma en una variable indicadora binaria `HasCabin` (1 si tiene cabina asignada, 0 si es nula).
2.  **Ingeniería de Características:**
    *   `FamilySize` = `SibSp` + `Parch` + 1.
    *   `IsAlone` = 1 si `FamilySize` == 1, de lo contrario 0.
3.  **Codificación de Variables:**
    *   Se realiza un mapeo binario para `Sex` (female = 0, male = 1).
    *   Se aplica One-Hot Encoding a la columna `Embarked` para evitar imponer una jerarquía artificial.
4.  **Escalamiento:**
    *   Se estandarizan las variables continuas `Age` y `Fare` utilizando un escalador estándar para mejorar la convergencia de la Regresión Logística y KNN.

#### Load (Carga)
Los datos preparados se dividen en un 70% para entrenamiento y un 30% para prueba, asegurando una validación robusta y previniendo el sobreajuste.

---

### Análisis Exploratorio de Datos (EDA)

Durante la fase de exploración visual se detectaron los siguientes patrones clave en el dataset:
*   **Efecto de Género:** Cerca del 74% de las mujeres a bordo sobrevivieron, en comparación con solo el 19% de los hombres, validando el impacto masivo de las normas sociales de la época.
*   **Efecto Socioeconómico (Pclass):** Los pasajeros de 1.ª clase tuvieron una tasa de supervivencia del 63%, frente a un 47% en 2.ª clase y solo el 24% en 3.ª clase, confirmando el acceso privilegiado a los recursos de evacuación.
*   **Efecto de la Edad:** Los niños pequeños (menores de 5 años) tuvieron tasas de supervivencia sustancialmente mayores dentro de sus respectivas clases socioeconómicas.

---

## Planificación y Desafíos de la Gestión de Datos

### Tabla 3: Desafíos Estratégicos de los Datos

| Factor | Fortalezas | Oportunidades | Debilidades | Amenazas |
| :--- | :--- | :--- | :--- | :--- |
| **Calidad de datos** | Formato tabular limpio. | Aplicar técnicas avanzadas de imputación. | Alta cantidad de valores nulos en la columna `Cabin`. | Imputaciones incorrectas de la edad pueden sesgar el modelo. |
| **Sesgo en el dataset** | Datos reales y documentados. | Estudiar el impacto del sesgo social en la IA. | Desbalance moderado en la supervivencia (61% fallecidos vs 39% sobrevivientes). | El modelo podría subestimar la supervivencia en condiciones extremas atípicas. |

---

### Tabla 4: Matriz DOFA – Gestión de Datos en el Proyecto Titanic

*   **Fortalezas (Internas):** Datos tabulares limpios y estructurados en formato CSV estándar. Reglas de supervivencia bien documentadas históricamente.
*   **Debilidades (Internas):** Datos faltantes en variables clave (edad y cabina). La muestra es limitada históricamente (no se pueden recolectar nuevos datos).
*   **Oportunidades (Externas):** Utilizar algoritmos explicables para educar sobre sesgos de clase y género en modelos predictivos.
*   **Amenazas (Externas):** Que los modelos de IA tiendan a sobrefavorcer el género femenino ignorando otras variables importantes debido a la alta correlación inicial.

---

## Objetivos y Dimensión Ética

### Objetivos SMART
1.  **Específico:** Desarrollar un modelo de clasificación binaria para predecir si un pasajero sobrevivió al siniestro del Titanic.
2.  **Medible:** Lograr una precisión (Accuracy) superior al 80% y un F1-Score mayor a 0.78 en el conjunto de prueba.
3.  **Alcanzable:** Construir el pipeline en Python usando Scikit-learn sobre el manifiesto de 891 pasajeros estructurado.
4.  **Relevante:** Demostrar de forma práctica la aplicación de pipelines CRISP-ML(Q) a problemas de clasificación con datos históricos reales.
5.  **Temporal:** Culminar y validar el proyecto en su totalidad antes del término académico fijado.

---

### Tabla 5: Consideraciones Éticas de la Plataforma

| Consideración Ética | Enfoque de Mitigación | ¿Aplica? |
| :--- | :--- | :--- |
| **Dignidad Humana e Identidad** | Los nombres individuales no se usan en el entrenamiento para evitar deshumanizar los perfiles. | Sí |
| **Transparencia Algorítmica** | Se proporcionan explicaciones del porqué de las predicciones a través de reglas lógicas del Árbol de Decisión. | Sí |
| **Sesgo Histórico Directo** | Se audita activamente el sesgo de clase y género, analizando críticamente el protocolo de evacuación. | Sí |

---

## Parte 2. Guía para el Elevator Pitch (5 minutos)

### Tabla 6: Estructura de la Presentación de 5 Minutos

| Tiempo | Contenido |
| :--- | :--- |
| **Minuto 0–1: El Problema** | Presentación del desastre del Titanic y el planteamiento de que la supervivencia no fue al azar, sino influida por variables estructurales y sociales. |
| **Minuto 1–2: La Solución** | Diseño de un modelo predictivo supervisado (Random Forest y Regresión Logística) para clasificar la supervivencia basándose en el perfil demográfico. |
| **Minuto 2–3: Los Datos** | Explicación del flujo ETL/EDA: imputación de edad por grupos, creación del tamaño de familia y análisis del impacto de la clase y el sexo. |
| **Minuto 3–4: La Ética** | Discusión sobre el sesgo histórico de clase y cómo mitigamos el uso no ético o estigmatizante de perfiles algorítmicos. |
| **Minuto 4–5: El Impacto** | Demostración de las aplicaciones construidas (Web local y Streamlit) para simular y educar de forma interactiva sobre la supervivencia. |

---

## Análisis Crítico y Estratégico del Equipo

### ¿Cuál es el mayor desafío en su proyecto?
El mayor desafío es el **sesgo histórico** y el **desbalance moderado**. Dado que el protocolo "mujeres y niños primero" se aplicó con rigidez en las cubiertas superiores, la variable `Sex` tiene una importancia predictiva tan alta que los clasificadores simples tienden a ignorar las demás variables. Un hombre joven de 1.ª clase que pagó una tarifa muy alta y tenía acceso preferencial a los botes podría ser erróneamente clasificado como "No sobreviviente" debido al peso abrumador del factor de género en el modelo lineal.

### ¿Cómo lo mitigarían estratégicamente?
Para mitigar esto, implementamos modelos de ensamble no lineales como **Random Forest** y ajustamos los pesos de clase (`class_weight='balanced'`). Esto obliga al optimizador a prestar atención a las combinaciones complejas de variables (por ejemplo, pasajeros masculinos de primera clase que viajaban acompañados) y reduce la tasa de falsos negativos en hombres con condiciones socioeconómicas favorables para la supervivencia.

---

## Referencias (APA 7.ª edición)

*   Frey, B. S., Savage, D. A., & Torgler, B. (2011). Behavior under extreme conditions: The Titanic disaster. *Journal of Economic Perspectives*, 25(1), 209–222. https://doi.org/10.1257/jep.25.1.209
*   Geron, A. (2019). *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2.ª ed.). O'Reilly Media.
*   Hall, P., & Gill, N. (2019). *An introduction to machine learning interpretability* (2.ª ed.). O'Reilly Media.
