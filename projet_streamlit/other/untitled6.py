import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Répertoire contenant les fichiers JSON
data_dir = r"C:\Users\niels\Git\waterlevelprediction\data"

# Fonction pour créer les lags
def create_lag_features(df, target_col, max_lag=14):
    df = df.copy()
    for i in range(1, max_lag + 1):
        df[f'lag{i}'] = df[target_col].shift(i)
    return df

# Fonction d'encodage
def label_encode_columns(data, columns):
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    return data

# Initialisation des résultats
resultats = {}

# Liste des fichiers JSON
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

# Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=5)
scoring = make_scorer(r2_score)

# Boucle sur chaque fichier
for file in tqdm(json_files):
    try:
        # Charger JSON
        with open(os.path.join(data_dir, file)) as f:
            fulljson = json.load(f)

        df = pd.DataFrame(fulljson["data"])
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d %H:%M")
        df = df.sort_values("datetime").reset_index(drop=True)

        # Encodage des variables catégorielles
        df = label_encode_columns(df, df.select_dtypes(include='object').columns)

        # Filtrage de l'incertitude
        df = df[df["associated_uncertainty"] <= 0.25]

        # Création des lags
        target = "orthometric_height_of_water_surface_at_reference_position"
        df = create_lag_features(df, target)
        df = df.dropna()

        # Features et target
        X = df[[f"lag{i}" for i in range(1, 15)]]
        y = df[target]

        scores = {}
        params = {}

        # LINEAR REGRESSION
        lr_model = LinearRegression()
        lr_scores = []
        for train_idx, test_idx in tscv.split(X):
            lr_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            score = r2_score(y.iloc[test_idx], lr_model.predict(X.iloc[test_idx]))
            lr_scores.append(score)
        scores["LinearRegression"] = np.mean(lr_scores)
        params["LinearRegression"] = {}

        # RANDOM FOREST
        rf = RandomForestRegressor(random_state=0)
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=tscv, scoring=scoring, n_jobs=-1)
        rf_grid.fit(X, y)
        scores["RandomForest"] = rf_grid.best_score_
        params["RandomForest"] = rf_grid.best_params_

        # XGBOOST
        xgb = XGBRegressor(random_state=0, verbosity=0)
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.05]
        }
        xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=tscv, scoring=scoring, n_jobs=-1)
        xgb_grid.fit(X, y)
        scores["XGBoost"] = xgb_grid.best_score_
        params["XGBoost"] = xgb_grid.best_params_

        # Sélection du meilleur modèle
        best_model = max(scores, key=scores.get)

        # Sauvegarde des résultats
        resultats[file] = {
            "best_model": best_model,
            "best_score": round(scores[best_model], 4),
            "model_params": params[best_model]
        }

    except Exception as e:
        print(f"Erreur avec le fichier {file}: {e}")

# Sauvegarde des résultats en JSON
with open("resultats_modeles.json", "w") as f:
    json.dump(resultats, f, indent=4)
