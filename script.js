/* ==========================================================================
   TITANIC SURVIVAL PREDICTOR — script.js
   Inferencia del Modelo y Actualización Interactiva de Componentes (Fuerzas y Cubiertas)
   ========================================================================== */

'use strict';

// --- Parámetros de Inferencia entrenados en Python ---
const coefs = [
    -0.9608686587510223,   // Pclass
    -1.2290355406379663,   // Sex (male=1, female=0)
    -0.6103508027437379,   // Age
    -0.497501467687198,    // SibSp
    -0.15135621331942808,  // Parch
    0.0905292698913328,    // Fare
    -0.295490025474795,    // IsAlone
    0.08950244679084764,   // Embarked_Q
    -0.09085077496460947   // Embarked_S
];

const intercept = -0.6542324595953298;

const means = [
    2.319422150882825,
    0.651685393258427,
    29.435666131621186,
    0.45746388443017655,
    0.37720706260032105,
    30.874003049759235,
    0.622792937399679,
    0.08507223113964688,
    0.7158908507223114
];

const stds = [
    0.830241984464629,
    0.47643629319357733,
    13.336207097254427,
    0.9637263760992566,
    0.8407330176008964,
    47.722071862808,
    0.48468741940013105,
    0.2789891514531871,
    0.45098906924048243
];

// --- SVG Gauge Setup ---
const circle = document.getElementById('progress-circle');
let circumference = 0;
if (circle) {
    const radius = circle.r.baseVal.value;
    circumference = radius * 2 * Math.PI;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = circumference;
}

// --- DOM Elements ---
const valAge = document.getElementById('val-age');
const valFare = document.getElementById('val-fare');
const valSibsp = document.getElementById('val-sibsp');
const valParch = document.getElementById('val-parch');

const sliderAge = document.getElementById('slider-age');
const sliderFare = document.getElementById('slider-fare');
const sliderSibsp = document.getElementById('slider-sibsp');
const sliderParch = document.getElementById('slider-parch');

const outcomeBadge = document.getElementById('outcome-badge');
const outcomeDesc = document.getElementById('prediction-desc');
const probText = document.getElementById('prob-percentage');

// Ticket Elements
const ticketAvatar = document.getElementById('ticket-avatar');
const ticketName = document.getElementById('ticket-name');
const ticketClass = document.getElementById('ticket-class');
const ticketFare = document.getElementById('ticket-fare');
const ticketEmbarked = document.getElementById('ticket-embarked');
const ticketFamily = document.getElementById('ticket-family');

// Decks Elements
const deck1 = document.getElementById('deck-1');
const deck2 = document.getElementById('deck-2');
const deck3 = document.getElementById('deck-3');

// Factor Elements
const fBarSex = document.getElementById('factor-bar-sex');
const fValSex = document.getElementById('factor-val-sex');
const fBarClass = document.getElementById('factor-bar-class');
const fValClass = document.getElementById('factor-val-class');
const fBarAge = document.getElementById('factor-bar-age');
const fValAge = document.getElementById('factor-val-age');
const fBarFamily = document.getElementById('factor-bar-family');
const fValFamily = document.getElementById('factor-val-family');

function updatePrediction() {
    // 1. Obtener valores de la UI
    const sex = parseInt(document.querySelector('input[name="sex"]:checked').value);
    const pclass = parseInt(document.querySelector('input[name="pclass"]:checked').value);
    const age = parseFloat(sliderAge.value);
    const fare = parseFloat(sliderFare.value);
    const sibsp = parseInt(sliderSibsp.value);
    const parch = parseInt(sliderParch.value);
    const embarkedVal = document.querySelector('input[name="embarked"]:checked').value;
    
    // Feature engineering
    const familySize = sibsp + parch + 1;
    const isAlone = familySize === 1 ? 1 : 0;
    const embQ = embarkedVal === 'Q' ? 1 : 0;
    const embS = embarkedVal === 'S' ? 1 : 0;

    // Actualizar visualizaciones de valores de sliders
    valAge.textContent = `${age} ${age === 1 ? 'año' : 'años'}`;
    valFare.textContent = `£ ${fare.toFixed(2)}`;
    valSibsp.textContent = sibsp;
    valParch.textContent = parch;

    // 2. Inferencia de Regresión Logística
    const rawFeatures = [pclass, sex, age, sibsp, parch, fare, isAlone, embQ, embS];
    const scaledFeatures = [];
    for (let i = 0; i < rawFeatures.length; i++) {
        scaledFeatures.push((rawFeatures[i] - means[i]) / stds[i]);
    }

    let z = intercept;
    for (let i = 0; i < coefs.length; i++) {
        z += coefs[i] * scaledFeatures[i];
    }

    const probSurvival = 1 / (1 + Math.exp(-z));
    const survived = probSurvival > 0.5;

    // 3. Actualizar UI de Resultados y Gauge
    probText.textContent = `${(probSurvival * 100).toFixed(1)}%`;
    if (circle) {
        const offset = circumference - (probSurvival * circumference);
        circle.style.strokeDashoffset = offset;
    }

    if (survived) {
        outcomeBadge.textContent = "SOBREVIVE";
        outcomeBadge.className = "outcome-badge outcome-survives";
        
        if (sex === 0) {
            outcomeDesc.textContent = "El pasajero es mujer. Las políticas de abordaje priorizaron a mujeres (tasa de supervivencia del 74.2%).";
        } else if (pclass === 1) {
            outcomeDesc.textContent = "Pasajero masculino de primera clase. La proximidad física a botes superiores posibilitó el rescate.";
        } else if (age < 12) {
            outcomeDesc.textContent = "El pasajero es un infante. Los protocolos prioritarios favorecieron a niños pequeños para salvarse.";
        } else {
            outcomeDesc.textContent = "Supervivencia estimada favorable debido al estatus social y al viaje acompañado.";
        }
    } else {
        outcomeBadge.textContent = "FALLECE";
        outcomeBadge.className = "outcome-badge outcome-dies";
        
        if (sex === 1 && pclass === 3) {
            outcomeDesc.textContent = "Perfil masculino de tercera clase. Representa el grupo con menor tasa de supervivencia (13.5%).";
        } else if (sex === 1) {
            outcomeDesc.textContent = "Pasajero adulto masculino. Sujeto a la política 'mujeres y niños primero' con mortalidad de 81.1%.";
        } else if (pclass === 3) {
            outcomeDesc.textContent = "Pasajero de tercera clase. La distancia de cubiertas de botes y barreras del barco limitaron su escape.";
        } else {
            outcomeDesc.textContent = "La avanzada edad del pasajero y su género redujeron las probabilidades de abordaje a botes.";
        }
    }

    // 4. Actualizar Pase de Abordaje (Boarding Pass)
    // Avatar dinámico
    if (age <= 10) {
        ticketAvatar.textContent = "👶";
        ticketName.textContent = sex === 0 ? "Niña pequeña" : "Niño pequeño";
    } else if (age >= 60) {
        ticketAvatar.textContent = sex === 0 ? "👵" : "👴";
        ticketName.textContent = sex === 0 ? "Adulto Mayor Fem" : "Adulto Mayor Masc";
    } else {
        ticketAvatar.textContent = sex === 0 ? "👩" : "👨";
        ticketName.textContent = sex === 0 ? "Adulto Femenino" : "Adulto Masculino";
    }

    ticketClass.textContent = `${pclass}ª Clase`;
    ticketFare.textContent = `£ ${fare.toFixed(2)}`;
    
    let portName = "Southampton";
    if (embarkedVal === 'C') portName = "Cherbourg";
    else if (embarkedVal === 'Q') portName = "Queenstown";
    ticketEmbarked.textContent = portName;

    ticketFamily.textContent = isAlone === 1 ? "1 (Solo)" : `${familySize} integrantes`;

    // 5. Actualizar Indicadores de Cubierta de Evacuación
    deck1.classList.remove('active-deck');
    deck2.classList.remove('active-deck');
    deck3.classList.remove('active-deck');
    
    if (pclass === 1) {
        deck1.classList.add('active-deck');
    } else if (pclass === 2) {
        deck2.classList.add('active-deck');
    } else if (pclass === 3) {
        deck3.classList.add('active-deck');
    }

    // 6. Actualizar Medidores de Factores (Fuerza Predictiva)
    // Normalizar influencia
    // Sex: Female (100% favorable), Male (15% favorable)
    const sexScore = sex === 0 ? 95 : 15;
    fBarSex.style.width = `${sexScore}%`;
    fBarSex.style.backgroundColor = sexScore > 50 ? 'var(--accent-green)' : 'var(--accent-red)';
    fValSex.textContent = sex === 0 ? "Femenino (+)" : "Masculino (-)";

    // Class: Pclass 1 (90%), 2 (55%), 3 (20%)
    const classScore = pclass === 1 ? 90 : (pclass === 2 ? 55 : 20);
    fBarClass.style.width = `${classScore}%`;
    fBarClass.style.backgroundColor = classScore > 50 ? 'var(--accent-green)' : 'var(--accent-red)';
    fValClass.textContent = `${pclass}ª Clase`;

    // Age: Niños (90%), Adultos jóvenes (60%), Adultos mayores (25%)
    let ageScore = 50;
    if (age < 10) ageScore = 88;
    else if (age > 55) ageScore = 22;
    else ageScore = 100 - (age * 1.1); // disminuye gradualmente
    ageScore = Math.max(Math.min(Math.round(ageScore), 95), 15);
    fBarAge.style.width = `${ageScore}%`;
    fBarAge.style.backgroundColor = ageScore > 50 ? 'var(--accent-green)' : 'var(--accent-red)';
    fValAge.textContent = `${age} años`;

    // Family Size: Solo (45%), Acompañado 2-4 (75%), Familia numerosa 5+ (15%)
    let famScore = 45;
    if (familySize > 1 && familySize <= 4) famScore = 80;
    else if (familySize > 4) famScore = 15;
    fBarFamily.style.width = `${famScore}%`;
    fBarFamily.style.backgroundColor = famScore > 50 ? 'var(--accent-green)' : 'var(--accent-red)';
    fValFamily.textContent = isAlone === 1 ? "Solo" : `${familySize} pers.`;
}

// --- Event Listeners ---
const inputs = document.querySelectorAll('input[type="radio"], input[type="range"]');
inputs.forEach(input => {
    input.addEventListener('input', updatePrediction);
    input.addEventListener('change', updatePrediction);
});

// Inicializar predicción
window.addEventListener('DOMContentLoaded', () => {
    updatePrediction();
});
