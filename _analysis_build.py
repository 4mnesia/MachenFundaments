# -*- coding: utf-8 -*-
"""Mejora el analisis de Fase 1 (1.7, 1.9) y Fase 2 (celdas triviales,
conclusion) y agrega la seccion 2.7 con varios graficos orientados a la
clasificacion, cada uno con su analisis."""
import json
PATH = 'MachenLearning.ipynb'
nb = json.load(open(PATH, encoding='utf-8'))
def s(c): return ''.join(c['source']) if isinstance(c['source'], list) else c['source']
def md(x): return {"cell_type": "markdown", "metadata": {}, "source": x}
def code(x): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": x}

# ---------- Reemplazos de markdown (Fase 1 y celdas triviales Fase 2) ----------
repl = {}

repl['### 1.7 Alcance y restricciones'] = """### 1.7 Alcance y restricciones

El alcance se limita al dataset historico disponible (rondas de CS:GO). No se incorporan variables externas como comunicacion del equipo o estrategia interna. La unidad minima de analisis y prediccion es el **equipo-ronda**. Los modelos describen el **resultado esperado de la ronda** (clasificacion, tarea principal) y la **ventaja economica inicial** (regresion) a partir de informacion pre-ronda; no pretenden anticipar las decisiones tacticas que ocurren durante la ronda."""

repl['### 1.9 Control de fuga de informacion'] = "### 1.9 Control de fuga de informacion"  # header igual

# cuerpo de 1.9 (cell de texto que empieza con "Para no contaminar")
repl['Para no contaminar la prediccion, se prohiben como predictoras todas las variables construidas con informacion de la ronda actual:'] = """Para no contaminar la prediccion se prohibe usar como predictoras la informacion que ocurre **durante o despues** de la ronda que se quiere predecir:

- Resultado de la ronda actual (`RoundWinner`) y del partido (`MatchWinner`).
- Kills, asistencias, headshots, supervivencia y granadas **de la ronda actual**.

En cambio, la **economia y la composicion de compra del inicio de la ronda** (equipamiento, tipo de armas) **si** son informacion pre-ronda valida para la **clasificacion**, porque la compra se decide antes de que la ronda se juegue. Esa misma economia actual se **excluye solo en la regresion**, ya que alli construye directamente el target `EquipmentAdvantage`.

Todo el historial (resultado, economia y bajas de rondas anteriores) se usa desplazado con `shift(1)`, de modo que cada fila contiene unicamente datos conocidos antes de iniciar la ronda."""

# celdas triviales Fase 2
repl['`df.head(10)` muestra las primeras 10 filas del dataframe, permitiéndonos ver cómo se ven los datos originales.'] = """**Estructura de los datos.** Las primeras filas confirman que la unidad original del dataset es el **jugador-ronda**: cada registro es la actuacion de un jugador en una ronda concreta de una partida (identificada por `MatchId` y `RoundId`). Esto ya anticipa una decision clave: como la compra es una decision **del equipo**, en la Fase 3 habra que **agregar la base a nivel equipo-ronda**."""

repl['`df.tail(10)` muestra las últimas 10 filas del dataframe, permitiéndonos ver cómo se ven los datos originales.'] = """**Consistencia al final del archivo.** Las ultimas filas mantienen la misma estructura, columnas y rangos que las primeras: no hay columnas corridas ni registros degenerados al final del archivo, por lo que el dataset se lee integro de principio a fin."""

repl['Aqui se expresan los tipos de datos de todas las variables del dataset'] = """Se revisan los tipos de cada columna, porque determinan el tratamiento posterior: las **numericas** necesitaran imputacion y escalado, y las **categoricas** necesitaran codificacion (Fase 3)."""

repl['Aquí podemos ver que el dataset se divide en variables del tipo: `Int`, `Object`, `Float` y `Bool`'] = """El dataset combina cuatro familias de tipos: **enteras** (conteos como kills, asistencias o numero de ronda), **flotantes** (economia y tiempos), **texto/categoricas** (`Map`, `Team`, `RoundWinner`) y **booleanas** (indicadores de tipo de arma). Esta mezcla obliga a un preprocesamiento **diferenciado por tipo**: imputacion + escalado para numericas y One-Hot Encoding para categoricas, que es justamente lo que arma el `ColumnTransformer` de la Fase 3."""

repl['Recorremos el dataframe para convertir los datos categóricos ordinales de `Int64` a `category` para poder analizar el dataset de forma más eficiente'] = """Se convierten las categoricas ordinales de `Int64` a `category` para que el analisis y el encoding posterior las traten como categorias y no como numeros con orden aritmetico (evitando que el modelo asuma distancias falsas entre niveles)."""

repl['Esto nos dice que hay *79157* **observaciones** y *30* **dimensiones**'] = """Con **79.157 filas** a nivel jugador-ronda y **30 variables**, el dataset tiene volumen mas que suficiente para modelar. Tras agregar a equipo-ronda en la Fase 3 quedaran ~15.800 observaciones, una cifra comoda para entrenar y validar con separacion por partidas (`GroupShuffleSplit`)."""

applied = 0
for c in nb['cells']:
    if c['cell_type'] != 'markdown':
        continue
    txt = s(c).strip()
    for key, new in repl.items():
        if txt == key or txt.startswith(key):
            c['source'] = new
            applied += 1
            break

# ---------- Conclusion Fase 2 reescrita ----------
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and s(c).startswith('### Conclusión fase 2'):
        c['source'] = """### Conclusión fase 2 (Data understanding)

El analisis exploratorio se oriento a la tarea principal del Parcial 4 (predecir `RoundWinner`) sin perder de vista la regresion economica. Los hallazgos clave son:

- **Calidad y volumen:** 79.157 registros jugador-ronda (~15.800 equipo-ronda), practicamente sin nulos (3 celdas). Los "outliers" no son errores sino rondas reales (eco, force-buy, desempenos extremos), por lo que se tratan con capping/escalado robusto y **no** se eliminan.
- **Target balanceado (~50/50):** favorece la clasificacion y hace que la accuracy sea interpretable; el baseline trivial ronda 0.50.
- **El bando importa:** `Terrorist` gana mas que `CounterTerrorist`, y la ventaja **varia por mapa** (interaccion `Map`x`Team`). Ambas entran como predictoras.
- **La economia no es lineal:** invertir mas **no** garantiza ganar (las rondas ganadas no tienen mayor economia promedio). Esto anticipa que modelos no lineales (arboles, vecindad, margen) superaran a la regresion lineal y justifica categorizar la compra (`InvestmentLevel`).
- **Control de fuga, visto en los datos:** las variables de la **ronda actual** (kills, asistencias, supervivencia) separan muy bien victorias de derrotas, pero ocurren *durante* la ronda; por eso solo se usan **desplazadas** (`shift(1)`) como historial, nunca las actuales.

Sobre esta base, la Fase 3 construye la vista equipo-ronda, las variables historicas sin fuga y el preprocesamiento (imputacion, IQR Capping, escalado y encoding) con separacion train/test por partida."""
        applied += 1
        break

# ---------- Seccion 2.7: graficos orientados al modelado ----------
c271_md = md("""### 2.7 Analisis dirigido al modelado (clasificacion)

Las secciones anteriores exploraron el dataset de forma general. Aqui el analisis se enfoca en la **tarea principal del Parcial 4**: predecir si el equipo gana la ronda. Se arma una vista equipo-ronda de apoyo (`tr_eda`) y se estudian las variables que mejor **separan victorias de derrotas**, para anticipar que predictoras seran utiles y cuales son fuga.""")

c271_code = code("""import numpy as np
# Vista equipo-ronda de apoyo para la EDA de clasificacion
_rw = df['RoundWinner'].astype(str).str.strip().replace({'False4': 'False'}).map({'True': 1, 'False': 0})
tr_eda = (df.assign(_RW=_rw)
          .groupby(['MatchId', 'RoundId', 'Map', 'Team'], observed=True)
          .agg(Gana=('_RW', 'first'),
               Economia=('TeamStartingEquipmentValue', 'median'),
               Kills=('RoundKills', 'sum'),
               Headshots=('RoundHeadshots', 'sum'),
               Asistencias=('RoundAssists', 'sum'))
          .reset_index()
          .dropna(subset=['Gana']))
tr_eda['Gana'] = tr_eda['Gana'].astype(int)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
vc = tr_eda['Gana'].value_counts().sort_index()
axes[0].pie(vc.values, labels=['Pierde', 'Gana'], autopct='%1.1f%%',
            colors=['#e76f51', '#2a9d8f'], wedgeprops=dict(width=0.45))
axes[0].set_title('Balance del target (equipo-ronda)')
wr_team = tr_eda.groupby('Team')['Gana'].mean().sort_values()
axes[1].barh(wr_team.index, wr_team.values * 100, color='#457b9d')
axes[1].axvline(50, color='red', ls='--'); axes[1].set_title('Winrate por bando')
axes[1].set_xlabel('% de victorias')
wr_map = tr_eda.groupby('Map')['Gana'].mean().sort_values()
axes[2].barh(wr_map.index, wr_map.values * 100, color='#f4a261')
axes[2].axvline(50, color='red', ls='--'); axes[2].set_title('Winrate por mapa')
axes[2].set_xlabel('% de victorias')
plt.tight_layout(); plt.show()""")

c271_an = md("""**Analisis:** el target esta **balanceado (~50/50)**, ideal para clasificacion: la accuracy es interpretable y el baseline trivial ronda 0.50. Pero aparece una **asimetria por bando** (`Terrorist` gana mas que `CounterTerrorist`), asi que `Team` sera una predictora relevante. Por **mapa** los winrates se mantienen mas cerca del 50%: el mapa por si solo no decide la ronda, aunque si modula la ventaja de bando (ver el cruce siguiente).""")

c272_code = code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))
piv = tr_eda.pivot_table(index='Map', columns='Team', values='Gana', aggfunc='mean') * 100
sns.heatmap(piv, annot=True, fmt='.1f', cmap='RdYlGn', center=50, ax=axes[0],
            cbar_kws={'label': '% victorias'})
axes[0].set_title('Winrate (%) por Mapa y Bando')
sns.boxplot(data=tr_eda, x='Gana', y='Economia', ax=axes[1])
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Pierde', 'Gana'])
axes[1].set_title('Economia inicial por resultado')
axes[1].set_xlabel(''); axes[1].set_ylabel('Equipamiento inicial')
plt.tight_layout(); plt.show()""")

c272_an = md("""**Analisis:** el mapa de calor muestra que la **ventaja de bando depende del mapa** (hay combinaciones mapa-bando muy por encima y por debajo del 50%), lo que confirma una **interaccion `Map`x`Team`** util para el modelo. El boxplot revela el hallazgo contraintuitivo central: la economia inicial de las rondas **ganadas no es mayor** que la de las perdidas; sus distribuciones se solapan. Es decir, *gastar mas no asegura ganar* -> la relacion economia-resultado es **no lineal** y debe combinarse con contexto e historial.""")

c273_code = code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))
num = ['Kills', 'Headshots', 'Asistencias', 'Economia', 'RoundId']
corr_t = tr_eda[num + ['Gana']].corr()['Gana'].drop('Gana').sort_values()
colors = ['#e76f51' if v < 0 else '#2a9d8f' for v in corr_t.values]
axes[0].barh(corr_t.index, corr_t.values, color=colors)
axes[0].axvline(0, color='black'); axes[0].set_xlabel('Correlacion de Pearson')
axes[0].set_title('Correlacion de las variables con el resultado')
for g, col, lab in [(0, '#e76f51', 'Pierde'), (1, '#2a9d8f', 'Gana')]:
    sns.kdeplot(tr_eda.loc[tr_eda['Gana'] == g, 'Kills'], ax=axes[1],
                fill=True, color=col, label=lab, alpha=0.4)
axes[1].set_title('Distribucion de bajas por resultado')
axes[1].set_xlabel('Bajas del equipo en la ronda'); axes[1].legend()
plt.tight_layout(); plt.show()""")

c273_an = md("""**Analisis (clave metodologica):** las **bajas de la ronda** son, por lejos, lo mas correlacionado con ganar, y el KDE muestra que **separan casi perfectamente** victorias de derrotas. Pero esas bajas ocurren *durante* la ronda: usarlas para predecir el resultado seria **fuga de informacion**. Por eso en la Fase 3 solo se usan **desplazadas** (`PrevRoundKills`, `RollingKillsLast3`). La **economia** correlaciona debil/negativamente con ganar, reforzando que su efecto es indirecto y no lineal. Esto explica por que el modelo honesto se apoya en **economia + historial pre-ronda**, no en el desempeno actual.""")

c274_code = code("""plt.figure(figsize=(11, 5))
sns.histplot(tr_eda['Economia'], bins=50, color='#457b9d')
ymax = plt.ylim()[1]
for x, lab in [(11000, 'Eco / Force'), (22000, 'Full-buy')]:
    plt.axvline(x, color='red', ls='--')
    plt.text(x, ymax * 0.92, '  ' + lab, color='red')
plt.title('Distribucion de la economia inicial del equipo (zonas de compra)')
plt.xlabel('Equipamiento inicial del equipo'); plt.ylabel('Frecuencia')
plt.tight_layout(); plt.show()""")

c274_an = md("""**Analisis:** la economia inicial es **multimodal**: se distinguen grupos de rondas eco/force (compra baja) y de full-buy (compra alta), con valles entre ellos. Esto **justifica categorizar la compra** en `InvestmentLevel` (baja / media / alta) en la Fase 3, porque captura un comportamiento discreto de decision economica que una sola variable continua difumina.""")

nuevos = [c271_md, c271_code, c271_an, c272_code, c272_an, c273_code, c273_an, c274_code, c274_an]

# insertar antes de la Conclusion de Fase 2
out = []
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and s(c).startswith('### Conclusión fase 2'):
        out.extend(nuevos)
    out.append(c)
nb['cells'] = out

json.dump(nb, open(PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# compile check
bad = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            compile(s(c), f'<{i}>', 'exec')
        except SyntaxError as e:
            bad.append((i, e.msg))
print('markdown reemplazados:', applied, '| celdas nuevas:', len(nuevos), '| total:', len(nb['cells']))
print('celdas que no compilan:', bad)
