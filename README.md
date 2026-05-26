# MachenFundaments - Proyecto CRISP-DM CS:GO

Proyecto analitico de Fundamentos de Machine Learning basado en el caso de
Counter-Strike: Global Offensive. El notebook principal es
`MachenLearning.ipynb`.

## Objetivo del proyecto

Apoyar decisiones estrategicas de compra y gestion de recursos antes o al inicio
de una ronda de CS:GO. La problematica central es que un equipo debe decidir si
conviene comprar fuerte, forzar compra o ahorrar, pero esa decision se toma bajo
incertidumbre.

El analisis usa variables de mapa, bando, economia, armamento e historial previo
para estudiar desempeno competitivo a nivel equipo-ronda.

## Estructura CRISP-DM

El notebook sigue la metodologia CRISP-DM:

1. Business Understanding: define problematica, objetivos, alcance, KPI e
   hipotesis de negocio.
2. Data Understanding: revisa estructura, nulos, outliers, distribuciones,
   correlaciones, mapas, bandos y variables temporales/territoriales.
3. Data Preparation: transforma la base desde jugador-ronda hacia equipo-ronda,
   crea variables derivadas, separa `X` e `y`, codifica categoricas, imputa,
   escala y valida train/test.
4. Modeling: actualmente desarrolla regresion con tres modelos y lectura
   predictiva.
5. Evaluation: pendiente de desarrollo formal.
6. Deployment: pendiente de desarrollo formal.

## Dataset

El notebook carga el CSV desde GitHub:

`https://raw.githubusercontent.com/FabianMolinaa/CSV/refs/heads/main/Anexo%20ET_demo_round_traces_2022.csv`

La carpeta `Material de apoyo/` contiene la rubrica, clases y ejemplos del curso.
El dataset original no debe modificarse. Toda transformacion debe realizarse en
copias de trabajo dentro del notebook.

## Estado actual

La Fase 1, Fase 2 y Fase 3 estan desarrolladas. La Fase 4 tiene una version de
regresion muy trazable con:

- Target actual: `AvgTeamTimeAliveSec`.
- Unidad de analisis: equipo-ronda.
- Variables predictoras: contexto, economia, composicion de armas e historial
  pre-ronda.
- Split por `MatchId` usando `GroupShuffleSplit` para simular partidas no vistas.
- Modelos: regresion lineal, arbol de decision y random forest.
- Metricas: `MAE`, `MSE`, `RMSE`, `R2`.

Ultima lectura validada:

- Mejor modelo de regresion: regresion lineal.
- `RMSE_test`: 34.6166.
- Mejora vs baseline en RMSE: 7.83%.
- `R2_test`: -0.0119.
- `MAE_test` no supera al baseline.

Conclusion actual: la regresion con `AvgTeamTimeAliveSec` esta bien implementada
metodologicamente, pero no es predictivamente fuerte. Conviene evaluar un cambio
de target de regresion si el objetivo es maximizar capacidad predictiva.

## Estado frente a rubrica

Segun el material de apoyo revisado, el notebook esta bien encaminado, pero no
debe considerarse terminado mientras Fase 5 y Deployment sigan vacias.

Cumplido o bien avanzado:

- CRISP-DM visible.
- Fase 1 alineada a negocio.
- Fase 2 con EDA amplio, graficos, nulos, outliers, distribuciones y
  correlaciones.
- Fase 3 con preparacion, encoding, scaling, train/test y control de fuga.
- Fase 4 regresion con tres modelos, metricas y visualizaciones.

Pendiente para quedar alineado al 100%:

- Desarrollar **Fase 5: Evaluation** con comparacion final de modelos,
  interpretacion de `MAE`, `MSE`, `RMSE`, `R2` y justificacion del mejor modelo.
- Desarrollar **Fase 6: Deployment**. Actualmente el notebook dice
  `Fase 5: Deployment`; debe corregirse a `Fase 6: Deployment` si se quiere
  CRISP-DM estricto.
- Implementar un formulario o celda interactiva que permita ingresar un nuevo
  registro y generar una prediccion, tal como pide el material de Parcial 3.
- Decidir si se mantiene `AvgTeamTimeAliveSec` como target de regresion o si se
  cambia el foco, ajustando Fase 1, Fase 3 y Fase 4 en conjunto.

## Posibles cambios de foco

Opciones discutidas:

- Mantener `AvgTeamTimeAliveSec`: cumple target numerico continuo, pero predice
  debilmente con variables pre-ronda.
- Cambiar a `TeamRoundKills`: mejora predictiva y sigue midiendo desempeno
  competitivo, pero es conteo discreto.
- Cambiar a `TeamStartingEquipmentValue` usando solo contexto e historial previo:
  predice mucho mejor, pero cambia el foco hacia economia esperada y no hacia
  desempeno competitivo.

No usar targets economicos con variables economicas actuales directamente como
predictoras, porque eso genera fuga/circularidad.

## Como ejecutar

1. Crear o activar entorno Python.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Abrir `MachenLearning.ipynb` en Jupyter.
4. Ejecutar el notebook completo de arriba hacia abajo.

Para validar por consola:

```bash
python -m jupyter nbconvert --to notebook --execute MachenLearning.ipynb --inplace --ExecutePreprocessor.timeout=900
```

## Continuar en otro PC

Para continuar el trabajo en otro equipo:

1. Copiar o clonar la carpeta completa del proyecto.
2. Verificar que existan `MachenLearning.ipynb`, `Material de apoyo/`,
   `README.md`, `AGENTS.md`, `resumen.md` y `requirements.txt`.
3. Instalar dependencias con `pip install -r requirements.txt`.
4. Leer primero `AGENTS.md` y `resumen.md`.
5. Abrir el notebook y ejecutar desde el inicio.
6. No asumir rutas absolutas del PC anterior; trabajar con rutas relativas al
   proyecto.
7. No modificar archivos de `Material de apoyo/`.
8. Antes de cambiar el modelamiento, decidir el target final de regresion.

## Archivos importantes

- `MachenLearning.ipynb`: notebook principal.
- `Material de apoyo/`: clases, rubricas, datasets auxiliares y ejemplos.
- `requirements.txt`: dependencias base.
- `AGENTS.md`: reglas para continuar el trabajo con otra IA/agente.
- `resumen.md`: memoria resumida del estado actual y decisiones tomadas.
