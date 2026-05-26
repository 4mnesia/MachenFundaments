# Resumen del proyecto

Este archivo guarda la memoria practica del trabajo realizado para continuar el
proyecto en otro entorno o con otro agente.

Ultima actualizacion de memoria: 2026-05-26.

Si este proyecto se abre en otro PC, leer primero:

1. `README.md`
2. `AGENTS.md`
3. `resumen.md`
4. `MachenLearning.ipynb`

No depender de rutas absolutas del equipo anterior. Trabajar siempre desde la
raiz del proyecto y validar que exista la carpeta `Material de apoyo/`.

## Proyecto

- Notebook principal: `MachenLearning.ipynb`.
- Dataset principal: CSV de rondas de CS:GO cargado desde GitHub.
- Carpeta de referencia: `Material de apoyo/`.
- Metodologia obligatoria: CRISP-DM.
- Idioma del notebook: espanol.

## Instrucciones del usuario

El usuario pidio:

- Actuar como Senior Data Scientist experto en Machine Learning.
- Seguir estrictamente CRISP-DM.
- Basar decisiones en el material de apoyo y rubricas.
- Usar Markdown antes del codigo para justificar decisiones.
- Usar Markdown despues de tablas/graficos para interpretar.
- Evitar celdas largas y codigo mezclado.
- Mantener todo muy trazable.
- Separar claramente `X` e `y`.
- Analizar nulos, outliers, distribuciones y correlaciones.
- Advertir fuga de informacion.
- Codificar categoricas antes de modelar.
- Escalar numericas solo si corresponde.
- Mantener variables originales para auditoria.
- Usar variables temporales por periodo, no promedios globales sin contexto.
- Usar visualizaciones territoriales/geograficas cuando aplique.
- En nombres visibles de modelos, incluir tipo/subtipo y fecha/hora.
- No crear apartados innecesarios solo para el timestamp del modelo.
- No modificar el dataset original.
- Si algo no esta alineado con la rubrica, decirlo de forma clara.
- Si una mejora no funciona, comprobar que falta y comentarlo.
- Mantener Fase 2 cargada de graficos; en Fase 3 y Fase 4 usar graficos solo de
  preparacion, control y evaluacion.

## Problematica actual

Los equipos de CS:GO deben decidir antes o al inicio de una ronda si compran,
fuerzan o ahorran. Esa decision se toma con incertidumbre. Invertir mas no
garantiza ganar, porque tambien influyen mapa, bando, rival, composicion de
armas e historial reciente.

El foco de negocio es apoyar decisiones estrategicas de compra y gestion de
recursos.

## Fase 1

La Fase 1 fue ajustada para que no hable solo de clasificacion. Ahora reconoce:

- Problema principal: apoyar decisiones de compra.
- Clasificacion: estimar victoria/derrota con `RoundWinner_Num`.
- Regresion: estimar una medida numerica complementaria de desempeno.

Estado: alineada con Fase 3 y Fase 4.

## Fase 2

Contiene analisis exploratorio amplio:

- Estructura del dataset.
- Tipos de datos.
- Nulos.
- Outliers.
- Duplicidad.
- Distribuciones.
- Analisis por mapa.
- Analisis por bando.
- Variables temporales/periodos.
- Correlaciones.
- Lecturas de negocio.

Estado: desarrollada, aunque puede pulirse redaccion si se cambia el foco final
de regresion.

## Fase 3

Transforma de jugador-ronda a equipo-ronda.

Pasos principales:

- Limpieza de `RoundWinner`.
- Creacion de `RoundWinner_Num`.
- Conversion de `TimeAlive` a `TimeAlive_Sec`.
- Agregacion por `MatchId`, `RoundId`, `Map`, `Team`.
- Creacion de `AvgTeamTimeAliveSec`.
- Variables de economia propia/rival.
- Variables de armas.
- Variables temporales pre-ronda con `shift`.
- Separacion de predictoras y targets.
- Pipeline con imputacion, IQR capping, encoding y `RobustScaler`.
- Split por partida usando `GroupShuffleSplit`.

Features actuales antes de encoding:

- 3 categoricas.
- 13 numericas base.
- 11 temporales pre-ronda.
- Total: 27 predictoras antes de encoding.
- Total despues de encoding: 30 columnas.

## Fase 4 actual

Se desarrollo la subseccion de regresion.

Target actual:

- `AvgTeamTimeAliveSec`.

Modelos:

- Baseline con `DummyRegressor`.
- Regresion lineal.
- Arbol de decision.
- Random Forest.

Metricas:

- `MAE`.
- `MSE`.
- `RMSE`.
- `R2`.

Ultimos resultados validados:

| Modelo | MAE_test | RMSE_test | R2_test |
| :--- | ---: | ---: | ---: |
| Regresion lineal | 24.7258 | 34.6166 | -0.0119 |
| Random Forest | 24.8441 | 34.9418 | -0.0310 |
| Arbol de decision | 24.9528 | 35.3668 | -0.0563 |
| Baseline mediana | 21.6552 | 37.5588 | -0.1913 |

Lectura:

- La regresion lineal mejora el RMSE frente al baseline.
- No mejora el MAE frente al baseline.
- `R2_test` sigue negativo.
- El modelo funciona parcialmente, pero no es predictivamente fuerte.

Graficos agregados despues de la preocupacion del usuario:

- Fase 3:
  - distribucion de targets y familias de predictoras;
  - outliers principales antes del tratamiento;
  - validacion visual del split.
- Fase 4:
  - comparacion grafica de `RMSE_test`, `MAE_test`, `R2_test`;
  - real vs predicho;
  - residuos;
  - importancia de variables en Random Forest.

## Verificaciones adicionales hechas por chat

Se probaron alternativas fuera del notebook para analizar cambio de target.

Resultados aproximados:

- `AvgTeamTimeAliveSec`: debil, `R2 < 0`.
- `TeamRoundKills`: mejor, `R2` cercano a 0.37 y mejora de RMSE cercana a 21%.
- `TeamStartingEquipmentValue` usando solo contexto e historial previo:
  `R2` cercano a 0.65 y mejora de RMSE cercana a 41%.
- `TeamStartingEquipmentValue` usando variables de compra actual: casi perfecto,
  pero invalido por fuga/circularidad.
- `EquipmentSpread` y `AvgPlayerEquipmentValue` tambien pueden dar metricas casi
  perfectas si se usan variables relacionadas directamente, pero eso no es
  metodologicamente valido.

## Discusion pendiente

El usuario quiere un enfoque completamente predictivo. La preocupacion es que
`AvgTeamTimeAliveSec` no esta entregando un modelo suficientemente fuerte.

Opciones viables:

1. Mantener `AvgTeamTimeAliveSec`.
   - Ventaja: target numerico continuo, mas seguro para rubrica.
   - Desventaja: bajo poder predictivo.

2. Cambiar a `TeamRoundKills`.
   - Ventaja: mas predictivo y conectado con desempeno ofensivo.
   - Desventaja: es conteo discreto, no continuo puro.

3. Cambiar a `TeamStartingEquipmentValue` con features solo historicas y de
   contexto.
   - Ventaja: fuerte predictivamente.
   - Desventaja: cambia el foco a economia esperada, no desempeno competitivo.

Recomendacion actual:

- Si se prioriza rubrica de regresion continua: mantener `AvgTeamTimeAliveSec` y
  explicar limitaciones.
- Si se prioriza poder predictivo y negocio ofensivo: cambiar a `TeamRoundKills`
  y justificarlo como variable numerica de impacto esperado.
- Si se reformula el negocio hacia anticipar economia: usar
  `TeamStartingEquipmentValue` con cuidado para evitar circularidad.

## Revision contra rubrica

Estado claro:

- Fase 1: alineada.
- Fase 2: alineada y visualmente fuerte.
- Fase 3: alineada, con preparacion, variables, encoding, scaling, split y
  control de fuga.
- Fase 4: alineada en regresion, tres modelos y metricas, pero el target actual
  no es predictivamente fuerte.
- Fase 5 Evaluation: falta desarrollarla formalmente.
- Deployment: falta desarrollarlo formalmente.
- El notebook tiene placeholder `## Fase 5: Deployment`; debe corregirse a
  `## Fase 6: Deployment` para CRISP-DM estricto.

Segun la rubrica de Parcial 3, para cierre final falta:

- evaluacion final de modelos de regresion;
- justificacion definitiva del mejor modelo;
- explicacion de combinacion de features;
- deployment/formulario de prediccion;
- notebook ejecutado y sin errores.

## Siguiente paso sugerido

Antes de seguir con clasificacion o evaluation, decidir target final de
regresion.

Si se cambia el target, ajustar:

- Fase 1: objetivo e hipotesis.
- Fase 3: tabla de targets, features y conclusiones.
- Fase 4: baseline, modelos, metricas, analisis y graficos.
- Fase 5: evaluacion final.

Si no se cambia el target, el siguiente paso es desarrollar Fase 5 con una
conclusion honesta: `AvgTeamTimeAliveSec` cumple con regresion continua, pero el
modelo tiene poder predictivo limitado. En ese caso, la utilidad principal del
proyecto deberia apoyarse mas en clasificacion o en decision de compra.

Si se cambia el target, hacerlo en cadena:

1. Reescribir objetivo de regresion en Fase 1.
2. Cambiar tabla de targets en Fase 3.
3. Ajustar features para evitar fuga.
4. Reentrenar Fase 4.
5. Volver a ejecutar el notebook completo.

## Comando de validacion

```bash
python -m jupyter nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=900
```

Ultima ejecucion completa del notebook: sin errores.
