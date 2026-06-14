# Resumen del proyecto

Memoria practica para continuar el proyecto en otro entorno o con otro agente.
Leer primero `README.md`, `AGENTS.md` y este archivo, luego `MachenLearning.ipynb`.

Ultima actualizacion: 2026-06-13 (v6, Parcial 4).

> **Correccion de datos importante:** la columna `RoundWinner` del dataset venia
> **invertida** (los registros marcados `True` matan menos -0.43 vs 0.91- y
> sobreviven menos -23% vs 58%-, es decir eran los **perdedores**). Se corrige en
> la celda de carga (`RoundWinner` -> `True` = gano). Esto NO cambia la magnitud de
> las metricas del modelo, pero corrige toda la interpretacion: con la etiqueta
> arreglada, mas economia/kills -> mas victorias (intuitivo), y el bando fuerte es
> `CounterTerrorist` (~57,6%), no `Terrorist`. No revertir este fix.

## Proyecto

- Notebook principal: `MachenLearning.ipynb` (CRISP-DM Fases 1 a 6).
- Dataset: rondas de CS:GO cargado desde GitHub (CSV, sep `;`).
- Metodologia: CRISP-DM. Idioma: espanol. Unidad de analisis: equipo-ronda.
- Entrega vigente: **Parcial 4 - Modelos de Clasificacion** (grupal, con
  presentacion, 30% de la nota).

## Que cubre la entrega Parcial 4

- **Fase 4 (Modeling):** 4.1 regresion (iteracion Parcial 3) y 4.2 clasificacion
  (tarea principal). Cada tarea con baseline + 3 modelos + metricas.
- **Fase 5 (Evaluation):** lectura de negocio del clasificador ganador (matriz de
  confusion, ROC-AUC, importancia de variables).
- **Fase 6 (Deployment):** `pickle` en la carpeta `modelos/` (un .pkl por modelo,
  nombre por tarea+algoritmo) + formulario interactivo `ipywidgets` que carga el
  ganador para predecir un registro nuevo.

## Tarea de clasificacion (principal)

- Target: `RoundWinnerNum` (`1` gana, `0` pierde). Balance ~50/50.
- Features: contexto (`Map`, `Team`, `InvestmentLevel`), momento (`RoundId`,
  `HasPreviousRound`), composicion de armas e historial pre-ronda (`Prev*`,
  `Rolling*`, incluidos `PrevRoundWin`, `PrevRoundKills`, `RollingKillsLast3`).
  Todo con `shift(1)`.
- Split por `MatchId` (`GroupShuffleSplit`). Preprocesamiento con `Pipeline` +
  `ColumnTransformer` (mediana + IQR Capping + `RobustScaler`; moda +
  `OneHotEncoder`).
- Modelos: `KNeighborsClassifier`, `RandomForestClassifier`, `SVC` (SVM, RBF) y un
  **ensamble por voto suave** (`VotingClassifier`). Baseline:
  `DummyClassifier(most_frequent)`. Cada modelo optimizado con `GridSearchCV` +
  `GroupKFold` (4.2.3) + metodo del codo para KNN. KNN y SVM dependen del escalado.
- Metricas: accuracy, precision, recall, F1, ROC-AUC (train y test) + matriz de
  confusion, ROC y Precision-Recall, con graficos de diagnostico por modelo.

Resultados (test): mejor modelo **Ensamble por voto suave**, ROC-AUC ~0.792,
accuracy ~0.698 (RF 0.788, KNN 0.786, SVM 0.776). El ensamble mejora marginalmente
a los individuales: la clasificacion esta en su **techo predictivo (~0.79
ROC-AUC)**, el tuning/escalado/features/ensamble no lo superan de forma drastica
(verificado). Supera con claridad al baseline. Nota: el ensamble pesa ~63 MB en
`.pkl`; el RF individual (~27 MB) es casi equivalente.

## Tarea de regresion (iteracion Parcial 3, seccion 4.1)

- Target: `EquipmentAdvantage = TeamStartingEquipmentValue - OpponentEquipmentValue`.
- **Control de fuga:** las predictoras se restringen a contexto + historial
  pre-ronda (`reg_categorical_features`, `reg_numeric_features`). Se EXCLUYEN las
  variables derivadas de la economia/composicion de la ronda actual
  (`LogEquipmentRatio`, `EquipmentSpread`, `*Share`, `*Delta*`, `InvestmentLevel`).
  Se INCLUYEN `PrevRoundWin` (resultado de la ronda anterior) y el historial de
  bajas (`PrevRoundKills`, `RollingKillsLast3`), predictores clave.
- Modelos: `LinearRegression`, `RandomForestRegressor`, `SVR`. Baseline:
  `DummyRegressor(median)`. Cada modelo optimizado con `GridSearchCV`. El SVR usa
  `TransformedTargetRegressor`(`StandardScaler`) para escalar el target.
- Resultado (test): Random Forest **R2 ~0.74** (MAE ~5300), SVR **R2 ~0.71**,
  lineal ~0.47. Sin fuga. Una version anterior inflaba el R2 a ~0.99 por incluir
  variables derivadas del propio target; eso quedo corregido.

## Decisiones de diseno tomadas

- Conservar regresion (Parcial 3) y clasificacion (Parcial 4) en secciones
  separadas y limpias dentro de la Fase 4.
- Target de clasificacion: `RoundWinnerNum` (no se cambio).
- Deployment con `pickle` + formulario `ipywidgets` dentro del notebook.
- La Fase 4 estaba duplicada; se consolido en una sola con numeracion unica.

## Trabajo pendiente / ideas

- Hecho: cada modelo (reg y clf) se optimiza con `GridSearchCV` + `GroupKFold`
  (4.1.2 / 4.2.3). La Fase 5 agrega overfitting (train vs test), matrices/ROC
  comparadas, residuos, curvas de aprendizaje e importancia de variables.
- Opcional: si se amplia el deployment, exponer una app Flask (material 4.3.3).
- Opcional: `XGBoost`/`LightGBM` o `CalibratedClassifierCV` podrian aranar
  decimas en clasificacion, pero no romperan el techo ~0.79.
- La Fase 3 conserva una doble definicion historica de `crear_preprocesador`
  (celda simple shadow + celda IQR); funciona, pero podria unificarse.

## Validacion

```bash
python -m nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

Ultima ejecucion: `n_errors = 0`, notebook completo, carpeta `modelos/` generada
con todos los modelos y el ganador verificado contra el pipeline en memoria.

Nota tecnica: la optimizacion (`GridSearchCV`, `cross_val_score`, `VotingClassifier`,
`learning_curve`) usa `n_jobs=1` para evitar un fallo intermitente de joblib/loky
en Windows (limpieza de memmap) durante `nbconvert`. Los `RandomForest` mantienen
`n_jobs=-1` internamente (hilos, sin ese problema).
