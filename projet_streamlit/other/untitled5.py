import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from statsmodels.graphics.tsaplots import plot_pacf

with open(rf"C:\Users\niels\Git\waterlevelprediction\data\Seine_Aube__2018-12-17__2025-03-03.json") as f:
    fulljson = json.load(f)
    
df_catalog=pd.DataFrame(fulljson["data"])

print(df_catalog.info())

df_catalog["datetime"] = pd.to_datetime(df_catalog["datetime"], format="%Y/%m/%d %H:%M")
categorical_colums = df_catalog.select_dtypes(include = ['object']).columns.tolist()

def label_encode_columns(data, columns):
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])        
    return data
df_traitement = label_encode_columns(df_catalog, categorical_colums)
df_traitement = df_traitement[df_traitement['associated_uncertainty'] <= 0.25]

print(df_traitement.info()) 


df_modelisation = df_traitement.sort_index()

# Trace de la PACF pour la série de niveau d’eau

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

plot_acf(df_modelisation['orthometric_height_of_water_surface_at_reference_position'],lags=40, ax=ax[0])
plot_pacf(df_modelisation['orthometric_height_of_water_surface_at_reference_position'],lags=40, ax=ax[1])
plt.show()
#quel jour precendent est t'il utile de prendre pour les predictions 

#on prend les quatres premiers car ils sont au dessus de la bande bleue ( haute correlation )
df_modelisation['lag0'] = df_modelisation['orthometric_height_of_water_surface_at_reference_position'].shift(0) #la meleur que on essaye de predire 

for i in range(1, 4):  # 14 lags
    df_modelisation[f'lag{i}'] = df_modelisation['orthometric_height_of_water_surface_at_reference_position'].shift(i)




#### LINEAR REGRESSION ##################################

from sklearn.linear_model import LinearRegression

# Supprimer les lignes avec des NaN (à cause du shift)
df_modelisation = df_modelisation.dropna()
X = df_modelisation[['lag1', 'lag2','lag3']] 
y =df_modelisation['orthometric_height_of_water_surface_at_reference_position']#la meleur que on essaye de predire 

model = LinearRegression()
model.fit(X, y)

import matplotlib.pyplot as plt

# Prédictions
y_pred = model.predict(X)

# Affichage comparatif
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Valeur réelle', color='blue')
plt.plot(y.index, y_pred, label='Valeur prédite', color='orange')
plt.title("Valeur réelle vs Valeur prédite du niveau d’eau")
plt.xlabel("Date")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


##### RANDOM FOREST #########################################################

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,      # plus d'arbres
    max_depth=15,          # plus de profondeur
    min_samples_split=5,   # plus de robustesse
    min_samples_leaf=2,
    random_state=0
)
model.fit(X, y)

y_pred = model.predict(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Valeur réelle', color='blue')
plt.plot(y.index, y_pred, label='Valeur prédite', color='orange')
plt.title("Valeur réelle vs Valeur prédite (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#### Random Forest Reg ???? #######################################################

#validation croisée temporelle 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Créer les lags de 1 à 14
for i in range(1, 15):
    df_modelisation[f'lag{i}'] = df_modelisation['orthometric_height_of_water_surface_at_reference_position'].shift(i)

# Supprimer les lignes avec des NaN (créés par les décalages)
df_modelisation = df_modelisation.dropna()


X = df_modelisation[[f'lag{i}' for i in range(1, 15)]]
y = df_modelisation['orthometric_height_of_water_surface_at_reference_position']
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

rmses = []
r2s = []

# Pour stocker les derniers résultats
y_test_last = None
y_pred_last = None

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Sauvegarder les prédictions du dernier fold pour l'affichage
    y_test_last = y_test
    y_pred_last = y_pred

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    rmses.append(rmse)
    r2s.append(r2)

    print(f"Fold RMSE: {rmse:.2f}, R²: {r2:.2f}")

print("\n--- Moyennes ---")
print(f"RMSE moyen: {np.mean(rmses):.2f}")
print(f"R² moyen: {np.mean(r2s):.2f}")


plt.figure(figsize=(12, 6))
plt.plot(y_test_last.index, y_test_last, label='Valeur réelle', color='blue')
plt.plot(y_test_last.index, y_pred_last, label='Valeur prédite (dernier fold)', color='orange')
plt.title("Valeur réelle vs Valeur prédite (Random Forest - Dernier fold)")
plt.xlabel("Date")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


###### XGBOOST ##################################################################################

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
model.fit(X, y)

y_pred = model.predict(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Valeur réelle', color='blue')
plt.plot(y.index, y_pred, label='Valeur prédite (XGBoost)', color='orange')
plt.title("Valeur réelle vs Valeur prédite du niveau d’eau (XGBoost)")
plt.xlabel("Date")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


################ ARIMA ######################################################################

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Série à modéliser
y = df_modelisation['orthometric_height_of_water_surface_at_reference_position']

# Créer et ajuster un modèle ARIMA(p,d,q)
# Exemple : p=3 (lags), d=1 (différence), q=2 (moyenne mobile)
model = ARIMA(y, order=(3, 1, 2))
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Prédiction
y_pred = model_fit.predict(start=y.index[1], end=y.index[-1], typ='levels')  # d=1 → commencer à 1

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(y, label="Valeur réelle")
plt.plot(y_pred, label="Prédiction ARIMA", color='orange')
plt.legend()
plt.title("Prédiction du niveau d’eau avec ARIMA")
plt.grid(True)
plt.tight_layout()
plt.show()


################## SARIMA ####################################################################

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Exemple : SARIMA(3,1,2)(1,1,1,12)
model = SARIMAX(y, order=(3, 1, 2), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Prédictions
y_pred = model_fit.predict(start=y.index[13], end=y.index[-1], typ='levels')  # car d=1, D=1, S=12

# Affichage
plt.figure(figsize=(12, 6))
plt.plot(y, label="Valeur réelle")
plt.plot(y_pred, label="Prédiction SARIMA", color='green')
plt.legend()
plt.title("Prédiction du niveau d’eau avec SARIMA")
plt.grid(True)
plt.tight_layout()
plt.show()

