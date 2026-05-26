# AGENTS.md

Guia para cualquier agente o colaborador que continue este proyecto.

Este archivo debe ser leido antes de editar el notebook. Su objetivo es que otro
agente, incluso en otro PC, pueda continuar con las mismas instrucciones,
criterios y memoria de trabajo.

## Rol esperado

Actuar como Senior Data Scientist experto en Machine Learning. El trabajo debe
ser trazable, analitico, claro y alineado con CRISP-DM. El usuario prefiere
explicaciones precisas, sin redundancia y con codigo fragmentado en celdas
pequenas.

## Reglas de estilo del notebook

- Escribir en espanol.
- Mantener la estructura CRISP-DM.
- Antes de cada bloque de codigo, explicar en Markdown que se hara y por que.
- Despues de tablas o graficos, agregar una lectura analitica breve.
- Evitar celdas gigantes de codigo.
- No mezclar limpieza, transformacion, modelamiento y evaluacion en una sola
  celda.
- Separar explicitamente variables predictoras (`X`) y variables objetivo (`y`).
- Mantener variables originales para auditoria cuando se creen variables
  derivadas.
- No modificar el CSV original ni los archivos de `Material de apoyo/`.
- Usar copias de trabajo como `df_prep`, `team_round` o `model_data`.
- No depender de rutas absolutas del PC anterior.
- Usar nombres de modelos visibles con tipo/subtipo y fecha/hora, como
  `Regresion - Lineal | YYYY-MM-DD HH:MM`.
- No crear apartados decorativos para explicar el timestamp; solo incluirlo en
  el nombre del modelo.
- Si se agregan graficos, deben tener titulo claro y lectura posterior. Fase 2
  puede tener muchos graficos; Fase 3 y Fase 4 deben tener graficos de control,
  preparacion o evaluacion, no EDA repetido.

## Material de apoyo que debe considerarse

La carpeta `Material de apoyo/` contiene clases y rubricas. Las mas relevantes
para el estado actual son:

- `Instrucciones Evaluacion Parcial 2 - Analisis exploratorio.pdf`
- `Instrucciones Evaluacion Parcial 3 - Modelos de Regresion.pdf`
- `2.2.1 Proyecto de ML.pptx`
- `2.2.2 Analisis Exploratorio.pptx`
- `2.3.1 Preparacion de Datos I.pptx`
- `2_4_2_Proyecto_Completo.ipynb`
- `3.1.1 Regresion.pptx`
- `3.1.2 Regresion Multiple y Metricas.pptx`
- `3_1_3_Regresion_Lineal_I.ipynb`
- `3_1_4_REGRESION_LINEAL_II_.ipynb`
- `3.4.1 Arboles_I.pptx`
- `3.4.3 Arboles_II.pptx`

La rubrica de Parcial 3 exige:

- CRISP-DM.
- Target numerico continuo para regresion.
- Train/test.
- Al menos tres modelos de regresion.
- Metricas `R2`, `MAE`, `MSE`, `RMSE`.
- Justificacion del mejor modelo.
- Deployment posterior.

Estado frente a rubrica:

- Fases 1, 2 y 3 estan bien desarrolladas.
- Fase 4 regresion esta bien trazada, con tres modelos y metricas.
- Falta desarrollar Fase 5 Evaluation.
- Falta desarrollar Deployment como Fase 6. Actualmente existe placeholder
  `Fase 5: Deployment`; corregir a `Fase 6: Deployment` antes de cierre final.
- Falta formulario o mecanismo interactivo para ingresar un nuevo registro y
  generar prediccion.

## Problematica de negocio

En CS:GO, los equipos deben decidir al inicio de cada ronda si conviene comprar
fuerte, forzar compra o ahorrar. Esa decision depende de economia, mapa, bando,
armamento, rival e historial reciente.

Pregunta central:

> Bajo que condiciones economicas, tacticas y contextuales conviene invertir en
> una ronda, y cuando el desempeno esperado es suficientemente bajo como para
> justificar ahorro o ronda eco?

## Estado metodologico actual

El notebook trabaja a nivel equipo-ronda, no jugador-ronda.

Targets actuales:

- Clasificacion: `RoundWinner_Num`.
- Regresion actual: `AvgTeamTimeAliveSec`.

Variables predictoras actuales:

- Contexto: `Map`, `Team`, `RoundId`.
- Economia: `TeamStartingEquipmentValue`, `OpponentEquipmentValue`,
  `EquipmentAdvantage`, `LogEquipmentRatio`, `InvestmentLevel`,
  `EquipmentSpread`.
- Armas: `AssaultRifleShare`, `SniperRifleShare`, `HeavyShare`, `SMGShare`,
  `PistolShare`, `RifleAdvantage`, `PistolDisadvantage`.
- Historial pre-ronda: `PrevRoundWin`, `PrevAvgTeamTimeAliveSec`,
  `RollingWinrateLast3`, `RollingSurvivalLast3`, cambios de economia y ventaja
  previa.

Importante: las variables temporales deben usar `shift`. No usar informacion de
la ronda actual ni futura para construir predictores historicos.

## Control de fuga de informacion

No usar como predictoras para modelos pre-ronda:

- `RoundWinner` o `RoundWinner_Num`.
- `AvgTeamTimeAliveSec`, `TimeAlive`, `TimeAlive_Sec`, `Survived`.
- `RoundKills`, `RoundAssists`, `RoundHeadshots`, `RoundFlankKills`.
- Granadas lanzadas durante la ronda si el objetivo es predecir antes de la
  ronda.
- `MatchWinner` o acumulados posteriores si no se puede justificar su
  disponibilidad temporal.
- Cualquier variable actual usada para predecir un target derivado de esa misma
  variable.

## Estado de Fase 4

La regresion actual esta implementada con:

- `DummyRegressor` como referencia.
- `LinearRegression`.
- `DecisionTreeRegressor`.
- `RandomForestRegressor`.
- `Pipeline` y `ColumnTransformer`.
- Split por `MatchId` con `GroupShuffleSplit`.

Resultado actual:

- Mejor modelo: regresion lineal.
- `RMSE_test`: 34.6166.
- Mejora de RMSE vs baseline: 7.83%.
- `R2_test`: -0.0119.
- `MAE_test` no supera al baseline.

Lectura: la implementacion es correcta, pero el target `AvgTeamTimeAliveSec` no
esta siendo suficientemente predictivo con variables disponibles antes de la
ronda.

## Decision pendiente importante

El usuario quiere algo mas predictivo. Se discutio cambiar el target de
regresion.

Opciones:

1. Mantener `AvgTeamTimeAliveSec`: mas seguro para la rubrica de target continuo,
   pero debil predictivamente.
2. Cambiar a `TeamRoundKills`: mas predictivo y alineado con desempeno ofensivo,
   pero es conteo discreto.
3. Cambiar a `TeamStartingEquipmentValue` usando solo contexto e historial
   previo: fuerte predictivamente, pero cambia el foco hacia economia esperada.

Si se cambia el target, ajustar Fase 1, Fase 3 y Fase 4 para que todo quede
alineado. No cambiar solo Fase 4.

Decision recomendada al continuar:

- Si se quiere maxima seguridad con la rubrica, mantener
  `AvgTeamTimeAliveSec`, porque es numerico continuo, y explicar que su poder
  predictivo es limitado.
- Si el usuario prioriza poder predictivo, evaluar cambiar a `TeamRoundKills`,
  dejando claro que es conteo numerico y que puede ser discutible como
  "continuo".
- No usar targets economicos con variables economicas actuales como predictoras.
  Eso produce metricas artificialmente perfectas por circularidad.

## Trabajo pendiente sugerido

Orden recomendado para seguir:

1. Confirmar target final de regresion con el usuario.
2. Si cambia el target, actualizar Fase 1, Fase 3 y Fase 4.
3. Desarrollar Fase 5 Evaluation:
   - tabla final de modelos;
   - interpretacion de `MAE`, `MSE`, `RMSE`, `R2`;
   - seleccion del mejor modelo;
   - limitaciones y utilidad de negocio;
   - conclusion ejecutiva.
4. Desarrollar Fase 6 Deployment:
   - seleccionar modelo final;
   - crear ejemplo de nuevo registro;
   - crear formulario simple o celda interactiva;
   - generar prediccion;
   - documentar variables requeridas.
5. Ejecutar notebook completo y verificar `n_errors = 0`.

## Forma de trabajar

- Leer primero `resumen.md`, `README.md` y este archivo.
- Revisar el notebook antes de editar.
- Hacer cambios pequenos y verificables.
- Ejecutar el notebook completo cuando se modifique modelamiento o preparacion.
- Si una mejora no funciona, documentar el hallazgo y explicar que falta.
- No perseguir metricas artificiales con fuga de informacion.
- Si el usuario pregunta si algo esta bien, responder con criterio: decir que si,
  que no, o que falta, sin adornar.
- Mantener la respuesta y el notebook alineados con el material de apoyo.

## Comando de validacion

```bash
python -m jupyter nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=900
```

Despues de ejecutar, revisar que no existan outputs de error.
