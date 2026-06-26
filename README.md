# MachenFundaments — Proyecto CRISP-DM (CS:GO)

Proyecto analitico de **Fundamentos de Machine Learning** sobre el caso de
Counter-Strike: Global Offensive. Notebook principal: `MachenLearning.ipynb`.

Corresponde a la **Evaluacion Parcial 4 — Modelos de Clasificacion** (30% de la
nota) y cubre las seis fases de CRISP-DM. La tarea principal es de
**clasificacion** (predecir si el equipo gana la ronda); la tarea de **regresion**
del Parcial 3 (`EquipmentAdvantage`) se conserva como **iteracion previa** dentro
de la Fase 4, segun el indicador 3.1.2 de la rubrica (mostrar mejora entre
entregas).

---

## Estructura del proyecto

```
MachenFundaments/
├── MachenLearning.ipynb     # Notebook principal (CRISP-DM Fases 1-6)
├── README.md                # Esta portada
├── AGENTS.md                # Guia para continuar con otra IA/agente
├── resumen.md               # Memoria del estado y decisiones
├── requirements.txt         # Dependencias
├── limpiar_modelos.cmd      # Borra los .pkl de modelos/ (liberar espacio)
├── modelos/                 # Modelos de clasificacion (.pkl), generados en la Fase 6
├── enunciados/              # PDFs de la evaluacion (Parcial 4 y encargo)
└── Material de apoyo/       # Clases, rubricas y ejemplos del curso (no se modifica)
```

La carpeta `modelos/` se genera en la Fase 6 con un `.pkl` por modelo de
clasificacion (`clasificacion_knn`, `clasificacion_random_forest`,
`clasificacion_svm`, `clasificacion_ensamble_voto_suave`). El ganador (el
ensamble) es el que se recarga y usa en la demo y el formulario. El modelo de
regresion no se serializa por su tamano (arboles sin podar); se reproduce
ejecutando la Fase 4.1. Para borrar los `.pkl` y liberar espacio se ejecuta
`limpiar_modelos.cmd`.

---

## Como ejecutar

1. Crear o activar el entorno e instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Abrir `MachenLearning.ipynb` en Jupyter/VS Code y ejecutar de arriba hacia
   abajo (Run All). La Fase 6 genera la carpeta `modelos/` y muestra el
   formulario interactivo (requiere `ipywidgets`).

3. Validar por consola (debe terminar sin errores):

   ```bash
   python -m nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=2400
   ```

> Nota: el tuning de SVR/SVM usa `n_jobs=1` (los `GridSearchCV`/`cross_val_score`)
> para evitar un fallo intermitente de joblib en Windows; los `RandomForest`
> mantienen `n_jobs=-1` internamente. Por eso la ejecucion completa tarda
> ~10-15 min.

---

## Demostracion (mostrar que el modelo funciona)

La demostracion esta **integrada en el notebook** (Fase 6), reutilizando el
modelo recargado desde el `.pkl` y los datos ya en memoria:

- **Fase 6.4** predice sobre **6 rondas aleatorias** del conjunto de prueba
  (cambian en cada ejecucion) y compara el resultado real con la prediccion.
- **Fase 6.3** ofrece un **formulario interactivo** (`ipywidgets`): se cambia una
  entrada y la prediccion responde en vivo.

---

## Objetivo y problema de negocio

Apoyar la decision de compra al inicio de una ronda de CS:GO (comprar fuerte,
forzar o ahorrar) bajo incertidumbre. La tarea principal es **predecir si el
equipo gana o pierde la ronda** usando solo informacion disponible antes de que
comience.

- **Target de clasificacion:** `RoundWinnerNum` (`1` gana, `0` pierde).
- **Unidad de analisis:** equipo-ronda.
- **Predictoras:** contexto (mapa, bando, nivel de inversion), momento de la
  partida e historial pre-ronda desplazado con `shift(1)`.
- **Split:** `GroupShuffleSplit` por `MatchId` (evalua en partidas no vistas).
- **Preprocesamiento:** `Pipeline` + `ColumnTransformer` (imputacion, IQR
  Capping, `RobustScaler`, `OneHotEncoder`).

---

## Estructura del notebook (CRISP-DM)

1. **Fase 1 — Business Understanding:** problematica, objetivos, variables
   objetivo, diccionario de datos, hipotesis y control de fuga.
2. **Fase 2 — Data Understanding:** estructura, nulos, outliers, duplicados,
   graficos y correlaciones; seccion 2.7 de EDA dirigida a la clasificacion.
3. **Fase 3 — Data Preparation:** base equipo-ronda, variables historicas,
   separacion `X`/`y`, preprocesamiento y splits de regresion y clasificacion
   (ambos por `MatchId`).
4. **Fase 4 — Modeling:** 4.1 regresion (`LinearRegression`,
   `RandomForestRegressor`, `SVR`) y 4.2 clasificacion (`KNeighborsClassifier`,
   `RandomForestClassifier`, `SVC` + **ensamble por voto suave**). Cada tarea:
   baseline + modelos **optimizados con `GridSearchCV` + `GroupKFold`** +
   metricas (train y test) + graficos por modelo + **metodo del codo** (KNN).
5. **Fase 5 — Evaluation:** overfitting (train vs test), matrices de confusion,
   curvas ROC y Precision-Recall comparadas, residuos de regresion, curvas de
   aprendizaje, importancia de variables y validacion de hipotesis.
6. **Fase 6 — Deployment:** serializacion con `pickle` + formulario interactivo.

---

## Resultados (ultima ejecucion validada)

**Clasificacion** (test), ordenado por ROC-AUC:

| Modelo                        | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| :---------------------------- | -------: | --------: | -----: | -----: | ------: |
| Ensamble voto suave (ganador) |   0.6981 |    0.6999 | 0.6938 | 0.6968 |  0.7916 |
| Random Forest                 |   0.6962 |    0.6977 | 0.6925 | 0.6951 |  0.7884 |
| KNN (k=41, distance)          |   0.6953 |    0.6974 | 0.6900 | 0.6937 |  0.7863 |
| SVM (SVC, C=1, RBF)           |   0.6916 |    0.6971 | 0.6775 | 0.6872 |  0.7763 |
| Baseline (clase mayoritaria)  |   0.5000 |    0.5000 | 1.0000 | 0.6667 |  0.5000 |

Mejor modelo: el **ensamble por voto suave** (KNN + RF + SVM), que promedia
probabilidades y supera marginalmente a los individuales. Todos superan con
claridad al baseline (0.50). Las tres familias quedan muy cerca entre si: la
clasificacion esta en su **techo predictivo** (~0.79 ROC-AUC), porque el
resultado de una ronda conserva aleatoriedad irreducible no observable antes de
su inicio (verificado con tuning, escalado y features adicionales).

**Regresion** — iteracion Parcial 3 (test), target `EquipmentAdvantage`:

| Modelo                  |   MAE |  RMSE |    R2 |
| :---------------------- | ----: | ----: | ----: |
| Random Forest (mejor)   |  5300 |  7080 | 0.736 |
| SVR (target escalado)   |  5350 |  7365 | 0.714 |
| Regresion Lineal        |  8279 | 10017 | 0.471 |
| Baseline (mediana)      | 10720 | 13776 | 0.000 |

El Random Forest logra **R2 ~0.74 sin fuga** combinando el resultado de la ronda
anterior (`PrevRoundWin`), el historial de bajas y la optimizacion por
`GridSearchCV`. El **SVR** llega a R2 ~0.71 gracias a escalar el target con
`TransformedTargetRegressor` (sin ese paso colapsa a ~0.30).

---

## Metodologia: control de fuga de informacion

Para la **clasificacion** (`RoundWinnerNum`) se usan solo variables disponibles
antes de la ronda: contexto, economia inicial (la decision de compra) e historial
desplazado con `shift(1)`. **No** se usan kills, asistencias, supervivencia ni
granadas de la ronda actual, ni `MatchWinner`.

Para la **regresion** (`EquipmentAdvantage`) el control es mas estricto: como el
target ES una cantidad economica de la ronda actual, las predictoras se restringen
a **contexto e historial pre-ronda** (`Map`, `Team`, `RoundId`, `PrevRoundWin`,
valores y medias moviles de rondas anteriores). Se **excluyen** las variables
derivadas de la economia/composicion de la ronda actual (`LogEquipmentRatio`,
`EquipmentSpread`, `*Share`, `*Delta*`, `InvestmentLevel`), que construyen el
target. Con este conjunto honesto, el R2 (~0.74) es real, frente al R2 inflado
(~0.99) de una version anterior que incluia variables derivadas del propio target.

### Correccion de datos (etiqueta invertida)

Durante la EDA se detecto que la columna `RoundWinner` del dataset venia
**invertida**: los registros marcados `True` mataban menos (0.43 vs 0.91 por
jugador) y sobrevivian menos (23% vs 58%), es decir, eran los **perdedores**. Se
corrige en la celda de carga (`True` = gano). Esto no cambia la magnitud de las
metricas, pero corrige toda la interpretacion: con la etiqueta arreglada, mas
economia/kills se asocian a mas victorias (intuitivo), y el bando mas fuerte es
`CounterTerrorist` (~57,6%).

---

## Dataset

El notebook carga el CSV directamente desde GitHub:

`https://raw.githubusercontent.com/FabianMolinaa/CSV/refs/heads/main/Anexo%20ET_demo_round_traces_2022.csv`

El dataset original no se modifica; toda transformacion ocurre en copias de
trabajo (`df_prep`, `team_round`, `model_data`). La carpeta `Material de apoyo/`
contiene la rubrica, clases y ejemplos del curso.
