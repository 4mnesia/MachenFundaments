# -*- coding: utf-8 -*-
"""Demostracion rapida del modelo desplegado (Parcial 4).

Carga el modelo ganador desde `modelos/` y predice sobre CASOS REALES del
conjunto de prueba (`modelos/ejemplos.csv`, generado por la Fase 6.4 del
notebook), comparando el resultado real de la ronda con la prediccion. Sirve
para mostrar en la presentacion que el artefacto entregado funciona, sin abrir
el notebook.

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

# --- 1) Cargar el modelo ganador (respaldo al Random Forest) ---
candidatos = ['clasificacion_ensamble_voto_suave.pkl', 'clasificacion_random_forest.pkl']
ruta = next((MODELOS / n for n in candidatos if (MODELOS / n).exists()), None)
if ruta is None:
    raise SystemExit('No hay modelos en modelos/. Ejecuta la Fase 6 del notebook.')
with open(ruta, 'rb') as f:
    modelo = pickle.load(f)

# --- 2) Cargar casos reales del conjunto de prueba ---
ej_path = MODELOS / 'ejemplos.csv'
if not ej_path.exists():
    raise SystemExit('Falta modelos/ejemplos.csv. Ejecuta la Fase 6.4 del notebook.')
ejemplos = pd.read_csv(ej_path)
y_real = ejemplos.pop('RoundWinnerNum_real')

# --- 3) Predecir y comparar con el resultado real ---
pred = modelo.predict(ejemplos)
proba = modelo.predict_proba(ejemplos)[:, 1]

print(f'Modelo cargado: {ruta.name}')
print('Demostracion con casos REALES del conjunto de prueba (no vistos en entrenamiento)\n')
print('{:<5}{:<11}{:<17}{:<9}{:<12}{:>7}'.format('Caso', 'Map', 'Team', 'Real', 'Prediccion', 'Prob.'))
print('-' * 62)
for k in range(len(ejemplos)):
    fila = ejemplos.iloc[k]
    r, p, pr = int(y_real.iloc[k]), int(pred[k]), float(proba[k])
    er = 'GANA' if r == 1 else 'PIERDE'
    ep = 'GANA' if p == 1 else 'PIERDE'
    estado = 'OK' if r == p else 'fallo'
    print('{:<5}{:<11}{:<17}{:<9}{:<12}{:>6.1%}  {}'.format(
        k + 1, str(fila['Map']), str(fila['Team']), er, ep, pr, estado))

aciertos = int((pred == y_real.values).sum())
print()
print(f'Aciertos: {aciertos}/{len(ejemplos)} casos. El modelo acierta ~70% de las rondas')
print('y aplica internamente el mismo preprocesamiento del entrenamiento (recibe datos crudos).')
