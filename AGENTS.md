# AGENTS.md

Guia para cualquier agente o colaborador que continue este proyecto. Leer antes
de editar el notebook, junto con `README.md` y `resumen.md`.

## Rol esperado

Actuar como Senior Data Scientist experto en Machine Learning. Trabajo trazable,
analitico, claro y alineado con CRISP-DM. El usuario prefiere explicaciones
precisas, sin redundancia, con codigo fragmentado en celdas pequenas y en
espanol.

## Alcance actual: Evaluacion Parcial 4 (Clasificacion)

La entrega vigente es el **Parcial 4 - Modelos de Clasificacion** (30% de la
nota, grupal y con presentacion). Cubre CRISP-DM Fases 4, 5 y 6:

- **Fase 4 - Modeling (clasificacion):** target categorico, train/test, al menos
  3 modelos de clasificacion, seleccion del mejor.
- **Fase 5 - Evaluation:** metricas derivadas de la matriz de confusion
  (accuracy, precision, recall, F1) + ROC-AUC, interpretadas en clave de negocio.
- **Fase 6 - Deployment:** aplicar el modelo en un entorno interactivo tipo
  formulario para predecir un registro nuevo.

La regresion del Parcial 3 (target `EquipmentAdvantage`) se **conserva** como
iteracion previa en la seccion 4.1 (indicador de rubrica 3.1.2: mostrar mejora
entre entregas).

## Reglas de estilo del notebook

- Escribir en espanol y mantener la estructura CRISP-DM (Fases 1 a 6).
- Antes de cada bloque de codigo, explicar en Markdown que se hara y por que.
- Despues de tablas o graficos, agregar una lectura analitica breve.
- Evitar celdas gigantes; no mezclar limpieza, transformacion, modelamiento y
  evaluacion en una sola celda.
- Numeracion de secciones consecutiva y unica (4.1, 4.2, ...). No duplicar
  encabezados ni funciones.
- Separar explicitamente `X` y `y`. Mantener variables originales para auditoria.
- No modificar el CSV original ni los archivos de `Material de apoyo/`.
- Usar copias de trabajo (`df_prep`, `team_round`, `model_data`).
- Rutas relativas, no absolutas. Reportar filas eliminadas en cada filtro.

## Estado metodologico actual

Unidad de analisis: **equipo-ronda**.

**Clasificacion (tarea principal Parcial 4):**

- Target: `RoundWinnerNum` (`1` gana, `0` pierde la ronda). Balance ~50/50.
  **Importante:** la columna `RoundWinner` del dataset venia INVERTIDA (los `True`
  eran los perdedores, verificado con kills y supervivencia); se corrige en la
  celda de carga (`True` = gano). No revertir ese fix.
- Predictoras: contexto (`Map`, `Team`, `InvestmentLevel`), momento (`RoundId`,
  `HasPreviousRound`) e historial pre-ronda (`Prev*`, `Rolling*`) + composicion
  de armas. Toda variable historica usa `shift(1)`.
- Preprocesamiento: `Pipeline` + `ColumnTransformer` (mediana + IQR Capping +
  `RobustScaler` para numericas; moda + `OneHotEncoder` para categoricas).
- Split por `MatchId` con `GroupShuffleSplit` (evalua en partidas no vistas).
- Modelos: `KNeighborsClassifier`, `RandomForestClassifier`, `SVC` (SVM, RBF) y un
  **ensamble por voto suave** (`VotingClassifier`, material 4.1.3) que promedia los
  tres. Baseline: `DummyClassifier(most_frequent)`. KNN y SVM usan `probability=True`
  y dependen del `RobustScaler` del pipeline.
- Tuning: cada modelo se optimiza con `GridSearchCV` + `GroupKFold` (4.2.3) +
  metodo del codo para KNN. El desempeno se mantiene en ~0.79 ROC-AUC (techo).
- Metricas: accuracy, precision, recall, F1, ROC-AUC (train y test) + matriz de
  confusion, curvas ROC y Precision-Recall. Graficos de diagnostico por modelo.

**Regresion (iteracion Parcial 3, seccion 4.1):**

- Target: `EquipmentAdvantage = TeamStartingEquipmentValue - OpponentEquipmentValue`.
- Control de fuga estricto: como el target es economia de la ronda actual, las
  predictoras se restringen a contexto + historial pre-ronda
  (`reg_categorical_features`, `reg_numeric_features`). Se EXCLUYEN
  `LogEquipmentRatio`, `EquipmentSpread`, `*Share`, `*Delta*` e `InvestmentLevel`
  (derivadas del propio target). Se INCLUYEN `PrevRoundWin` (resultado de la ronda
  anterior) y el historial de bajas (`PrevRoundKills`, `RollingKillsLast3`),
  predictores clave de la economia. Con `GridSearchCV` el Random Forest alcanza
  R2 honesto ~0.74 en test (MAE ~5300).
- Modelos: `LinearRegression`, `RandomForestRegressor`, `SVR`. Baseline:
  `DummyRegressor(median)`. El **SVR** va envuelto en `TransformedTargetRegressor`
  (`StandardScaler`) para escalar el target monetario; sin eso colapsa (~0.30),
  con eso llega a R2 ~0.71. Cada modelo se optimiza con `GridSearchCV`.

## Control de fuga de informacion

No usar como predictoras informacion de la ronda actual que ocurra durante o
despues de ella: `RoundKills`, asistencias, headshots, supervivencia, granadas,
`RoundWinner`/`MatchWinner` actuales. Solo historial desplazado y contexto.
Para la regresion, ademas, se excluyen las variables economicas/de composicion de
la ronda actual porque construyen el target (ver seccion anterior).

## Deployment (Fase 6)

- Serializacion de **todos** los modelos con `pickle` en la carpeta `modelos/`
  (un archivo por modelo, nombre por tarea+algoritmo). El ganador de clasificacion
  es el que se despliega en el formulario.
- Carga del `.pkl` ganador y verificacion contra el pipeline en memoria.
- Formulario interactivo con `ipywidgets` (campos `Map`, `Team`,
  `InvestmentLevel`, `RoundId`; el resto de features se completan con la
  mediana/moda del entrenamiento). Fallback con `input()` si no hay `ipywidgets`.

## Resultados actuales (clasificacion, test)

| Modelo                       |     F1 | ROC-AUC | Accuracy |
| :--------------------------- | -----: | ------: | -------: |
| Ensamble voto suave (gana)   | 0.6968 |  0.7916 |   0.6981 |
| Random Forest                | 0.6951 |  0.7884 |   0.6962 |
| KNN (k=41, distance)         | 0.6937 |  0.7863 |   0.6953 |
| SVM (SVC, C=1)               | 0.6872 |  0.7763 |   0.6916 |
| Baseline (clase mayoritaria) | 0.6667 |  0.5000 |   0.5000 |

Mejor modelo: ensamble por voto suave (KNN+RF+SVM), por ROC-AUC; supera marginalmente
al RF. Todos sobre el baseline; las familias quedan parejas (~0.79 = techo).
Regresion (iteracion Parcial 3): Random Forest R2 ~0.74 y SVR (target escalado)
R2 ~0.71. Overfitting: KNN `weights='distance'` da ROC train = 1.0 (memoriza) pero
test 0.79; el SVM es el menos sobreajustado (train 0.81). **Nota deploy:** el
ensamble pesa ~63 MB en `.pkl`; si se requiere ligero, serializar el RF (~27 MB,
diferencia <0.005 ROC).

## Forma de trabajar

- Cambios pequenos y verificables. Ejecutar el notebook completo cuando se
  modifique preparacion o modelado.
- No perseguir metricas artificiales con fuga de informacion.
- Si el usuario pregunta si algo esta bien, responder con criterio: si, no o que
  falta, sin adornar.

## Comando de validacion

```bash
python -m nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=1800
```

Despues de ejecutar, revisar que no existan outputs de error.
