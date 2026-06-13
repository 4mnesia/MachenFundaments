# MachenFundaments - Proyecto CRISP-DM CS:GO

Proyecto analitico de Fundamentos de Machine Learning sobre el caso de
Counter-Strike: Global Offensive. El notebook principal es
`MachenLearning.ipynb`.

## Alcance actual (Parcial 4)

La entrega vigente corresponde a la **Evaluacion Parcial 4 - Modelos de
Clasificacion** (30% de la nota). Cubre las fases de CRISP-DM:

- **Fase 4 - Modeling (clasificacion)**: target categorico, train/test, al
  menos tres modelos de clasificacion y seleccion del mejor.
- **Fase 5 - Evaluation**: lectura de negocio de las metricas de la matriz de
  confusion y ROC-AUC, importancia de variables del modelo ganador.
- **Fase 6 - Deployment**: serializacion del modelo con `pickle` y formulario
  interactivo (`ipywidgets`) para predecir un registro nuevo.

La tarea de **regresion** desarrollada en el Parcial 3 (target
`EquipmentAdvantage`) se conserva como **iteracion previa** dentro de la Fase 4
(seccion 4.1), de acuerdo con el indicador 3.1.2 de la rubrica (mostrar mejora
entre entregas).

## Objetivo del proyecto

Apoyar la decision de compra al inicio de una ronda de CS:GO (comprar fuerte,
forzar o ahorrar) bajo incertidumbre. La tarea principal del Parcial 4 es de
clasificacion: **predecir si el equipo gana o pierde la ronda** usando solo
informacion disponible antes de su inicio.

- Target de clasificacion: `RoundWinnerNum` (`1` gana, `0` pierde).
- Unidad de analisis: equipo-ronda.
- Predictoras: contexto (mapa, bando, nivel de inversion), momento de partida e
  historial pre-ronda desplazado con `shift(1)`.
- Split: `GroupShuffleSplit` por `MatchId` (evalua en partidas no vistas).
- Preprocesamiento: `Pipeline` + `ColumnTransformer` (imputacion, IQR Capping,
  `RobustScaler`, `OneHotEncoder`).

## Estructura del notebook

1. **Fase 1 - Business Understanding**: problematica, objetivo, target e
   hipotesis de negocio.
2. **Fase 2 - Data Understanding**: estructura, nulos, calidad, distribuciones,
   correlaciones.
3. **Fase 3 - Data Preparation**: base equipo-ronda, variables historicas,
   separacion `X`/`y`, preprocesamiento y splits de **regresion y
   clasificacion** (ambos por `MatchId`).
4. **Fase 4 - Modeling**: 4.1 regresion (`LinearRegression`,
   `RandomForestRegressor`, `SVR`) y 4.2 clasificacion (`KNeighborsClassifier`,
   `RandomForestClassifier`, `SVC` + **ensamble por voto suave**). Cada tarea:
   baseline + modelos **optimizados con `GridSearchCV` + `GroupKFold`** + metricas
   (train y test) + graficos por modelo + **metodo del codo** (KNN).
5. **Fase 5 - Evaluation**: overfitting (train vs test), matrices de confusion,
   ROC y **Precision-Recall** comparadas, residuos de regresion, curvas de
   aprendizaje, importancia de variables, **validacion de hipotesis** y margen de mejora.
6. **Fase 6 - Deployment**: `pickle` + formulario interactivo.

## Resultados (ultima ejecucion validada)

Clasificacion (test), ordenado por ROC-AUC:

| Modelo                       | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| :--------------------------- | -------: | --------: | -----: | ----: | ------: |
| Ensamble voto suave (ganador)|   0.6981 |    0.6999 | 0.6938 | 0.6968 |  0.7916 |
| Random Forest                |   0.6962 |    0.6977 | 0.6925 | 0.6951 |  0.7884 |
| KNN (k=41, distance)         |   0.6953 |    0.6974 | 0.6900 | 0.6937 |  0.7863 |
| SVM (SVC, C=1, RBF)          |   0.6916 |    0.6971 | 0.6775 | 0.6872 |  0.7763 |
| Baseline (clase mayoritaria) |   0.5000 |    0.5000 | 1.0000 | 0.6667 |  0.5000 |

Mejor modelo: **Ensamble por voto suave** (KNN + Random Forest + SVM), que promedia
probabilidades y supera marginalmente a los modelos individuales (ROC-AUC 0.792).
Todos superan con claridad al baseline (0.50). Aun asi, las familias quedan muy
cerca entre si: la clasificacion esta en su **techo predictivo** (~0.79 ROC-AUC),
porque el resultado de una ronda conserva aleatoriedad irreducible no observable
antes de su inicio. Se incluyen ademas el **metodo del codo** (seleccion de `k`
para KNN) y **curvas Precision-Recall**.

Regresion - iteracion Parcial 3 (test), target `EquipmentAdvantage`:

| Modelo                       |   MAE |  RMSE |    R2 |
| :--------------------------- | ----: | ----: | ----: |
| Random Forest (mejor)        |  5300 |  7080 | 0.736 |
| SVR (target escalado)        |  5350 |  7365 | 0.714 |
| Regresion Lineal             |  8279 | 10017 | 0.471 |
| Baseline (mediana)           | 10720 | 13776 | 0.000 |

El Random Forest de regresion logra **R2 ~0.74 sin fuga** combinando el resultado
de la ronda anterior (`PrevRoundWin`), el historial de bajas (`PrevRoundKills`,
`RollingKillsLast3`) y la optimizacion por `GridSearchCV`. El **SVR** salta a
R2 ~0.71 gracias a escalar el target con `TransformedTargetRegressor` (sin ese
paso colapsa a ~0.30).

## Dataset

El notebook carga el CSV desde GitHub:

`https://raw.githubusercontent.com/FabianMolinaa/CSV/refs/heads/main/Anexo%20ET_demo_round_traces_2022.csv`

La carpeta `Material de apoyo/` contiene la rubrica, clases y ejemplos del
curso. El dataset original no se modifica; toda transformacion ocurre en copias
de trabajo (`df_prep`, `team_round`, `model_data`).

## Control de fuga de informacion

Para la clasificacion (`RoundWinnerNum`) se usan solo variables disponibles
antes de la ronda: contexto, economia inicial (decision de compra) e historial
desplazado con `shift(1)`. No se usan kills, asistencias, supervivencia ni
granadas de la ronda actual, ni `MatchWinner`.

Para la regresion (`EquipmentAdvantage`) se aplica un control de fuga mas
estricto: como el target ES una cantidad economica de la ronda actual, su
conjunto de predictoras se restringe a **contexto e historial pre-ronda**
(`Map`, `Team`, `RoundId`, resultado de la ronda anterior `PrevRoundWin`,
valores y medias moviles de rondas anteriores). Se **excluyen** las variables
derivadas de la economia o composicion de la ronda actual (`LogEquipmentRatio`,
`EquipmentSpread`, `*Share`, `*Delta*`, `InvestmentLevel`), que componen
directamente el target. Con este conjunto honesto (mas el historial de bajas y
tuning), el Random Forest logra **R2 ~0.74 en test** (clave: el resultado de la
ronda previa determina gran parte de la economia actual), frente al R2 inflado
(~0.99) de una version anterior que incluia variables derivadas del propio
target.

## Como ejecutar

1. Crear o activar entorno Python e instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Abrir `MachenLearning.ipynb` en Jupyter y ejecutar de arriba hacia abajo.
3. La Fase 6 genera la carpeta `modelos/` con todos los modelos serializados y
   muestra el formulario interactivo (requiere `ipywidgets`).

Validar por consola:

```bash
python -m nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

## Estructura del proyecto

```
MachenFundaments/
├── MachenLearning.ipynb     # Notebook principal (CRISP-DM Fases 1-6)
├── demo.py                  # Demostracion por consola: carga el modelo y predice casos reales
├── README.md                # Esta portada
├── AGENTS.md                # Guia para continuar con otra IA/agente
├── resumen.md               # Memoria del estado y decisiones
├── requirements.txt         # Dependencias
├── modelos/                 # Modelos de clasificacion (.pkl) + ejemplos.csv, Fase 6
├── enunciados/              # PDFs de la evaluacion (Parcial 4 y encargo)
└── Material de apoyo/       # Clases, rubricas y ejemplos del curso (no se modifica)
```

Para una **demostracion rapida** de que el modelo funciona (sin abrir el
notebook): `python demo.py` carga el modelo ganador desde `modelos/` y predice
sobre casos reales del conjunto de prueba, comparando el resultado real con la
prediccion. Dentro del notebook, la **Fase 6.4** hace lo mismo y la **Fase 6.3**
ofrece un formulario interactivo.

La carpeta `modelos/` contiene un `.pkl` por modelo de clasificacion, con nombre
descriptivo (`clasificacion_knn`, `clasificacion_random_forest`,
`clasificacion_svm` y `clasificacion_ensamble_voto_suave`); el ganador (el
ensamble) es el que carga el formulario de la Fase 6. El modelo de regresion no se
serializa por su tamano (arboles sin podar); se reproduce ejecutando la Fase 4.1.
