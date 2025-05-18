import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.model_selection import TimeSeriesSplit ,  GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


folder_path = r"C:\Users\niels\Git\waterlevelprediction\projet_streamlit\data"
max_filtered_lines = 0
file_with_max_filtered_lines = None

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict) and "data" in content:
                df_temp = pd.DataFrame(content["data"])
                df_temp['associated_uncertainty'] = pd.to_numeric(
                    df_temp['associated_uncertainty'], errors='coerce'
                )
                df_filtered_temp = df_temp[df_temp['associated_uncertainty'] <= 0.25* df_temp['orthometric_height_of_water_surface_at_reference_position']].copy()
                df_filtered_temp = df_filtered_temp.drop_duplicates(subset=['datetime']).copy()
                n_filtered_lines = len(df_filtered_temp)
                if n_filtered_lines > max_filtered_lines:
                    max_filtered_lines = n_filtered_lines
                    file_with_max_filtered_lines = file


print(f"\n Fichier avec le plus de lignes après filtrage (uncertainty <= 0.25) et suppression des doublons:")
print(f" - {file_with_max_filtered_lines}: {max_filtered_lines} lignes")


with open(rf"C:\Users\niels\Git\waterlevelprediction\data\{file_with_max_filtered_lines}", "r", encoding="utf-8") as f:
    fulljson = json.load(f)
print(fulljson.keys())
print(fulljson.values())

df_catalog=pd.DataFrame(fulljson["data"]) 


print(df_catalog.info()) #collones ne type objects a chnager 
print(df_catalog.isnull().sum()) #0


df_catalog["datetime"] = pd.to_datetime(df_catalog["datetime"], format="%Y/%m/%d %H:%M") #changement du format de la date pour l'utiliser
categorical_colums = df_catalog.select_dtypes(include = ['object']).columns.tolist()

def label_encode_columns(data, columns): #encoding des autres valerus meme si peu utilise a la prédiction
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])        
    return data
df_traitement = label_encode_columns(df_catalog, categorical_colums)

df_traitement = df_traitement.rename(columns={'orthometric_height_of_water_surface_at_reference_position': 'water_height'}) #renommer pour plus de facilité d'untilisation
print(df_traitement.info()) 

df_visualisation = df_traitement.set_index('datetime')
df_visualisation.index = pd.to_datetime(df_visualisation.index)

# Créer les colonnes année et jour de l'année
df_visualisation['year'] = df_visualisation.index.year
df_visualisation['day_of_year'] = df_visualisation.index.dayofyear
df_visualisation['month'] = df_visualisation.index.month


#correlation matrix and trying to see if there is a correlation between the variables
corr=  df_visualisation.corr()
sns.heatmap(corr, annot=True,)
plt.show()
sns.scatterplot(x='water_height',y='satellite',data= df_traitement)
plt.show()
sns.scatterplot(x='time',y='satellite',data= df_traitement)
plt.show()




freq_df = df_visualisation.groupby(['year', 'month']).size().reset_index(name='count')

plt.figure(figsize=(12, 6))
sns.barplot(data=freq_df, x='month', y='count', hue='year', palette='tab10')

plt.title("Fréquence de mesure par mois (couleur par année)")
plt.xlabel("Mois")
plt.ylabel("Nombre de mesures")
plt.xticks(ticks=range(12), labels=[
    'Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
    'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'
])
plt.legend(title='Année', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5)) 
df_visualisation['water_height'].plot(figsize=(12, 5), title="Niveau d'eau dans le temps complet")
plt.show()

sns.histplot(df_traitement['water_height'])
plt.title("Distribution du niveau d’eau")
plt.show()

#les plus grandes hauteurs 
top10 = df_visualisation.sort_values(by='water_height', ascending=False).head(10)
print(top10)

#les plus petites hauteurs 
minus10 = df_visualisation.sort_values(by='water_height', ascending=True).head(10)
print(minus10)

# On calcule la moyenne pour chaque jour de l'année sur toutes les années
daily_avg = df_visualisation.groupby('day_of_year')['water_height'].mean()

# Visualisation
plt.figure(figsize=(12, 5))
plt.plot(daily_avg.rolling(7).mean())  # moyenne glissante sur 7 jours
plt.title("Niveau moyen de l'eau par jour de l'année (toutes années confondues)")
plt.xlabel("Jour de l'année (1 à 366)")
plt.ylabel("Hauteur orthométrique moyenne")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))

# Boucle sur les années pour avoir la moyenne 
for year in sorted(df_visualisation['year'].unique()):
    yearly_data = df_visualisation[df_visualisation['year'] == year]
    plt.plot(
        yearly_data['day_of_year'].values,
        yearly_data['water_height'].values,
        label=str(year)
    )


plt.title("Profil journalier du niveau d’eau par année")
plt.xlabel("Jour de l’année")
plt.ylabel("Hauteur orthométrique (m)")
plt.legend(title="Année", loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='month', y='water_height', data=df_visualisation)
plt.title("Boxplot du niveau d’eau par mois")

df_modelisation = df_traitement.sort_index()

# Trace de la PACF pour la série de niveau d’eau
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

plot_acf(df_modelisation['water_height'],lags=100, ax=ax[0])
plot_pacf(df_modelisation['water_height'],lags=100, ax=ax[1])
plt.show()
#quel jour précendent est il utile de prendre pour les predictions 


# 1. Créer les lags
lags = 5
df = df_modelisation.copy()

# Créer les lags
for i in range(1, lags + 1):
    df[f"lag{i}"] = df["water_height"].shift(i)

# Supprimer les NaN (causés par shift)
df.dropna(inplace=True)

# Créer X et y directement à partir du même DataFrame
X = df[[f"lag{i}" for i in range(1, lags + 1)]].copy()
y = df["water_height"].copy()

# Réinitialiser les index pour eviter le problemes de data qui ne sont pas de la même taille
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


LinReg=LinearRegression()
LinReg.fit(X_train, y_train)
y_pred_LinReg =LinReg.predict(X_test)


plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Valeur réelle', color='blue')
plt.plot(y_test.index, y_pred_LinReg, label='Valeur prédite', color='orange')
plt.title("Valeur réelle vs prédite du niveau d’eau regression")
plt.xlabel("Date" if isinstance(y_test.index[0], pd.Timestamp) else "Index temporel")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

RFReg = RandomForestRegressor(random_state=42)

#la grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 10, None], # profondeur de chaque arbre
    'min_samples_split': [2, 5],  # min obs pour diviser un noeud
    'max_features': ['sqrt', 'log2'] # nombre de features à choisir par arbre
}
#GridSearchCV (avec validation croisée à 3 splits)
grid_search = GridSearchCV(
    estimator=RFReg,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # ou 'r2', 'neg_mean_absolute_error', etc.
    cv=tscv,
    n_jobs=-1,
    verbose=1
)
# 4. Entraîner GridSearchCV
grid_search.fit(X_train, y_train)

# 5. Récupérer le meilleur modèle
best_RFReg = grid_search.best_estimator_

# 6. Prédire avec le modèle optimisé
y_pred_RFReg = best_RFReg.predict(X_test)

# 7. Afficher les résultats
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Valeur réelle', color='blue')
plt.plot(y_test.index, y_pred_RFReg, label='Valeur prédite (Random Forest optimisé)', color='orange')
plt.title("Valeur réelle vs prédite du niveau d’eau (Test set) - Random Forest optimisé")
plt.xlabel("Date" if isinstance(y_test.index[0], pd.Timestamp) else "Index temporel")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.8,1.0],
    'colsample_bytree': [0.8,1.0]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=0),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_XG = grid_search.best_estimator_
y_pred_XG = best_XG.predict(X_test)


plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Valeur réelle', color='blue')
plt.plot(y_test.index, y_pred_XG, label='Valeur prédite (XGBoost)', color="orange")
plt.title("Valeur réelle vs Valeur prédite du niveau d’eau (XGBoost)")
plt.xlabel("Date")
plt.ylabel("Niveau d’eau (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

importances = best_XG.feature_importances_
features = X_train.columns



def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Fonctions métriques
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f" - R² (train)  : {r2_score(y_train, y_train_pred):.4f}")
    print(f" - R² (test)   : {r2_score(y_test, y_test_pred):.4f}")
    print(f" - MSE (test)  : {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f" - MAE (test)  : {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f" - MAPE (test) : {mape(y_test, y_test_pred):.2f}%")
    
print("evaluation of linear regression model")
evaluate_model(LinReg, X_train, y_train, X_test, y_test)
print("evaluation of random forest regression model") 
evaluate_model(best_RFReg, X_train, y_train, X_test, y_test)
print("evaluation XG model") 
evaluate_model(best_XG, X_train, y_train, X_test, y_test)



############### ARIMA ####################

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Série à modéliser
y = df_modelisation['water_height']

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










