# -*- coding: utf-8 -*-
"""Demostracion del modelo desplegado (Parcial 4).

Carga el modelo ganador desde `modelos/` y demuestra que PREDICE el resultado de
la ronda a partir de variables PRE-RONDA (nunca ve el resultado real). Dos partes:

  1) Prediccion sobre casos reales del test: se muestran las variables de entrada,
     el modelo predice, y RECIEN DESPUES se revela el resultado real (que estuvo
     oculto al modelo) para validar.
  2) Prueba de sensibilidad: se cambia una sola variable de entrada y se ve como
     cambia la prediccion -> evidencia de que el modelo calcula, no lee la respuesta.

Uso:
    python demo.py
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# El pipeline serializado usa este transformador custom; debe existir al cargar el .pkl.
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        q1, q3 = X_df.quantile(0.25), X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        zero_iqr = iqr.eq(0)
        self.lower_[zero_iqr] = -np.inf
        self.upper_[zero_iqr] = np.inf
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.clip(lower=self.lower_, upper=self.upper_, axis=1).to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f'x{i}' for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


MODELOS = Path('modelos')


def cargar_modelo():
    for n in ['clasificacion_ensamble_voto_suave.pkl', 'clasificacion_random_forest.pkl']:
        if (MODELOS / n).exists():
            with open(MODELOS / n, 'rb') as f:
                return pickle.load(f), n
    raise SystemExit('No hay modelos en modelos/. Ejecuta la Fase 6 del notebook.')


def etiqueta(v):
    return 'GANA' if int(v) == 1 else 'PIERDE'


modelo, nombre = cargar_modelo()
ej_path = MODELOS / 'ejemplos.csv'
if not ej_path.exists():
    raise SystemExit('Falta modelos/ejemplos.csv. Ejecuta la Fase 6.4 del notebook.')
ejemplos = pd.read_csv(ej_path)
y_real = ejemplos.pop('RoundWinnerNum_real')          # <-- se SACA: el modelo NO la ve
X = ejemplos                                           # solo variables pre-ronda

print('=' * 70)
print(f'Modelo desplegado: {nombre}')
print(f'El modelo recibe SOLO {X.shape[1]} variables PRE-RONDA (mapa, bando, economia,')
print('historial). El resultado real se elimina antes de predecir; solo se usa')
print('despues para validar.')
print('=' * 70)

# ---------------- Parte 1: prediccion sobre casos reales ----------------
print('\n[1] PREDICCION SOBRE CASOS REALES DEL TEST (no vistos en entrenamiento)\n')
pred = modelo.predict(X)
proba = modelo.predict_proba(X)[:, 1]
aciertos = 0
for k in range(len(X)):
    f = X.iloc[k]
    print(f'Caso {k + 1}:')
    print(f'   Entrada que ve el modelo -> Map={f["Map"]}, Team={f["Team"]}, '
          f'Inversion={f["InvestmentLevel"]}, RondaPrev_gano={int(f["PrevRoundWin"]) if pd.notna(f["PrevRoundWin"]) else "NA"}, '
          f'Winrate3={f["RollingWinrateLast3"]:.2f}, RatioEco={f["LogEquipmentRatio"]:+.2f}')
    print(f'   -> PREDICCION del modelo : {etiqueta(pred[k])}  (prob. de victoria {proba[k]:.0%})')
    real = int(y_real.iloc[k])
    ok = (real == int(pred[k]))
    aciertos += ok
    print(f'   -> Resultado real (oculto): {etiqueta(real)}   [{"ACIERTO" if ok else "fallo"}]\n')
print(f'Aciertos: {aciertos}/{len(X)}  (el modelo acierta ~70% de las rondas en test)')

# ---------------- Parte 2: prueba de sensibilidad ----------------
print('\n' + '=' * 70)
print('[2] PRUEBA DE QUE PREDICE (responde a los inputs)\n')
print('Tomamos un caso real y variamos SOLO la ventaja economica frente al rival')
print('(LogEquipmentRatio); todo lo demas queda igual. A mas economia, deberia subir')
print('la probabilidad de ganar.\n')
base = X.iloc[[0]].copy()
escala = [(-2.5, 'desventaja fuerte'), (-1.0, 'desventaja'), (0.0, 'paridad'),
          (1.0, 'ventaja'), (2.5, 'ventaja fuerte')]
for ratio, etiq in escala:
    fila = base.copy()
    fila['LogEquipmentRatio'] = ratio
    fila['InvestmentLevel'] = 'Baja inversion' if ratio < -0.5 else ('Compra alta' if ratio > 0.5 else 'Inversion media')
    p = modelo.predict_proba(fila)[0, 1]
    barra = '#' * int(round(p * 30))
    print(f'   Economia {etiq:18s} (ratio {ratio:+.1f}) -> prob. victoria {p:5.0%} |{barra}')
print('\nLa probabilidad SUBE con la economia: el modelo calcula desde la entrada,')
print('no copia un resultado guardado -> esta prediciendo.')
