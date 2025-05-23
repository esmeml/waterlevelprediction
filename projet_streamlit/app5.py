import streamlit as st
import os
import json
import pydeck as pdk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.model_selection import TimeSeriesSplit ,  GridSearchCV


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings



st.set_page_config(page_title="Observations Hydrologiques", layout="wide")
st.title("Information sur les cours d’eau")

DOSSIER_JSON = r"C:\Users\niels\Git\waterlevelprediction\projet_streamlit\data"
#DOSSIER_JSON = r"C:\Users\noara\Git\waterlevelprediction\projet_streamlit\data"

# Lister les fichiers JSON
fichiers = [f for f in os.listdir(DOSSIER_JSON) if f.endswith(".json")]

# Charger les données dans un dictionnaire
stations = {}
coords_list = []
for fichier in fichiers:
    chemin = os.path.join(DOSSIER_JSON, fichier)
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
            river = data["properties"].get("river", fichier.replace(".json", ""))
            stations[river] = data
            lon, lat = data["geometry"]["coordinates"]
            coords_list.append({"river": river, "lat": lat, "lon": lon})
    except Exception as e:
        st.warning(f"Erreur lors du chargement de {fichier}: {e}")

# Créer DataFrame des coordonnées
df_coords = pd.DataFrame(coords_list)

# Sélection d'une station
choix = st.selectbox("Sélectionnez un cours d’eau :", ["-- Aucune sélection --"] + sorted(stations.keys()))

# Déterminer la vue initiale pour la carte
if choix != "-- Aucune sélection --":
    # Récupérer les données de la station sélectionnée
    data = stations[choix]
    coords = data["geometry"]["coordinates"]  # Long, Lat
    initial_view = pdk.ViewState(latitude=coords[1], longitude=coords[0], zoom=10)  # Zoom ajusté à 10
else:
    # Si aucune station n'est sélectionnée, centrer sur un point par défaut
    df_filtre = df_coords[(df_coords["lat"] >= 40) & (df_coords["lat"] <= 50)]

    if not df_filtre.empty:
        initial_view = pdk.ViewState(
            latitude=df_filtre["lat"].mean(),
            longitude=df_filtre["lon"].mean(),
            zoom=4
        )
    else:
        initial_view = pdk.ViewState(latitude=0, longitude=0, zoom=1)

# Affichage de la carte
st.subheader("Carte des stations")

if not df_coords.empty:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_coords,
        get_position='[lon, lat]',
        get_color='[0, 100, 200, 160]',
        get_radius=5000,
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=initial_view,
        layers=[layer],
        tooltip={"text": "{river}"}
    ))
else:
    st.warning("Aucune coordonnée disponible pour les stations.")

# Affichage des données de la station sélectionnée
if choix != "-- Aucune sélection --":
    props = data["properties"]
    mesures = data.get("data", [])

    
    st.subheader("Niveaux d’eau observés")
    if mesures:
        df = pd.DataFrame(mesures)
        df = df.rename(columns={'orthometric_height_of_water_surface_at_reference_position': 'height'})
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d %H:%M")
        df = df.sort_values("datetime")

        # Choix du mode de sélection
        mode_selection = st.selectbox("Mode de sélection :", ["Période personnalisée", "Année entière", "Saison"], key="mode")

        # Définition des bornes disponibles
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()
        df["year"] = df["datetime"].dt.year

        if mode_selection == "Période personnalisée":
            start_date, end_date = st.date_input(
                "Sélectionnez une période :",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        elif mode_selection == "Année entière":
            années = sorted(df["year"].unique())
            année_choisie = st.selectbox("Choisissez une année :", années)
            start_date = pd.to_datetime(f"{année_choisie}-01-01").date()
            end_date = pd.to_datetime(f"{année_choisie}-12-31").date()
        elif mode_selection == "Saison":
            saisons = {
                "Printemps (21 mars – 20 juin)": ("03-21", "06-20"),
                "Été (21 juin – 22 septembre)": ("06-21", "09-22"),
                "Automne (23 septembre – 20 décembre)": ("09-23", "12-20"),
                "Hiver (21 décembre – 20 mars)": ("12-21", "03-20")
            }
            saison_choisie = st.selectbox("Choisissez une saison :", list(saisons.keys()))
            années = sorted(df["year"].unique())
            année_choisie = st.selectbox("Choisissez une année :", années, key="annee_saison")

            debut, fin = saisons[saison_choisie]
            if saison_choisie == "Hiver (21 décembre – 20 mars)":
                start_date = pd.to_datetime(f"{année_choisie}-12-21").date()
                end_date = pd.to_datetime(f"{année_choisie + 1}-03-20").date()
            else:
                start_date = pd.to_datetime(f"{année_choisie}-{debut}").date()
                end_date = pd.to_datetime(f"{année_choisie}-{fin}").date()

        # Filtrage
        df_filtré = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]

        if df_filtré.empty:
            st.warning("Aucune donnée disponible pour la période sélectionnée.")
        else:
            import altair as alt

            # Calcul de l'intervalle dynamique pour l'axe Y
            y_min, y_max = df_filtré["height"].min(), df_filtré["height"].max()
            marge = (y_max - y_min) * 0.1
            
            # Graphique Altair
            chart = alt.Chart(df_filtré).mark_line(color='steelblue').encode(
                x=alt.X("datetime:T", title="Date"),
                y=alt.Y("height:Q", title="Hauteur d’eau", scale=alt.Scale(domain=[y_min - marge, y_max + marge])),
                tooltip=["datetime:T", "height:Q"]
            ).properties(
                width=700,
                height=300,
                title="Hauteur d’eau observée"
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Aucune donnée de mesure trouvée.")


    with st.expander("Données brutes (table)"):
        if 'df_filtré' in locals() and not df_filtré.empty:
            st.dataframe(df_filtré[["datetime", "height", "associated_uncertainty", "satellite"]])
        else:
            st.write("Aucune donnée à afficher.")
    
    with st.expander("Métadonnées complètes"):
        st.json(props)






def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100  # éviter division par 0

if choix != "-- Aucune sélection --" and 'df_filtré' in locals() and not df_filtré.empty:
    df_model = df_filtré.copy()
    df_model["datetime"] = pd.to_datetime(df_model["datetime"])
    df_model = df_model.sort_values("datetime")
    
    # Vérification du nombre de données par an
    nb_mesures_par_annee = df_model["datetime"].dt.year.value_counts()
    annees_insuffisantes = nb_mesures_par_annee[nb_mesures_par_annee < 12]
    if not annees_insuffisantes.empty:
        st.warning(
            f"Moins de 12 mesures pour les années suivantes : {', '.join(map(str, annees_insuffisantes.index))}. "
            "Les modèles entraînés sur ces années peuvent être peu fiables."
        )

    
    lags = 5
    for i in range(1, lags + 1):
        df_model[f"lag{i}"] = df_model["height"].shift(i)
    df_model.dropna(inplace=True)
    
    X = df_model[[f"lag{i}" for i in range(1, lags + 1)]]
    y = df_model["height"]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # LINEAR REGRESSION
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    # RANDOM FOREST
    rf = RandomForestRegressor()
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_rf.fit(X, y)
    y_pred_rf = grid_rf.predict(X)
    
    # XGBOOST
    xgb = XGBRegressor()
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_xgb.fit(X, y)
    y_pred_xgb = grid_xgb.predict(X)
    
    # ARIMA
    y_arima = df_model["height"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            model_arima = ARIMA_model(y_arima, order=(3, 1, 2))
            model_arima_fit = model_arima.fit()
            y_pred_arima = model_arima_fit.predict(start=1, end=len(y_arima)-1, typ="levels")
        except:
            y_pred_arima = [np.nan] * (len(y_arima)-1)

    # SARIMA
    try:
        model_sarima = SARIMAX(y_arima, order=(3, 1, 2), seasonal_order=(1, 1, 1, 12))
        model_sarima_fit = model_sarima.fit(disp=False)
        y_pred_sarima = model_sarima_fit.predict(start=13, end=len(y_arima)-1, typ="levels")
    except:
        y_pred_sarima = [np.nan] * (len(y_arima)-13)

    # METRICS
    metrics = {}

    def compute_metrics(y_true, y_pred, name):
        y_true = y_true[:len(y_pred)]  # aligner tailles
        return {
            "R²": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            
        }

    metrics["LinearRegression"] = compute_metrics(y, y_pred_lr, "LR")
    metrics["RandomForest"] = compute_metrics(y, y_pred_rf, "RF")
    metrics["XGBoost"] = compute_metrics(y, y_pred_xgb, "XGB")
    if not np.isnan(y_pred_arima).all():
        metrics["ARIMA"] = compute_metrics(y[1:], y_pred_arima, "ARIMA")
    if not np.isnan(y_pred_sarima).all():
        metrics["SARIMA"] = compute_metrics(y[13:], y_pred_sarima, "SARIMA")

    
    # Séparation manuelle train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    def compute_metrics(model, X_train, y_train, X_test, y_test):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
        return {
            "R² Train": r2_score(y_train, y_pred_train),
            "R² Test": r2_score(y_test, y_pred_test),
            "MSE": mean_squared_error(y_test, y_pred_test),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "MAE": mean_absolute_error(y_test, y_pred_test),
            
        }
    
    # Recalcul des métriques avec la séparation train/test
    metrics = {}
    metrics["LinearRegression"] = compute_metrics(lr, X_train, y_train, X_test, y_test)
    metrics["RandomForest"] = compute_metrics(grid_rf.best_estimator_, X_train, y_train, X_test, y_test)
    metrics["XGBoost"] = compute_metrics(grid_xgb.best_estimator_, X_train, y_train, X_test, y_test)
    
    # VotingRegressor
    voting = VotingRegressor(estimators=[
        ('lr', lr),
        ('rf', grid_rf.best_estimator_),
        ('xgb', grid_xgb.best_estimator_),
    ])
    voting.fit(X_train, y_train)
    
    
    # ARIMA / SARIMA (pas de split train/test ici)
    if not np.isnan(y_pred_arima).all():
        metrics["ARIMA"] = {
            "R² Train": "—", "R² Test": r2_score(y[1:], y_pred_arima),
            "MSE": mean_squared_error(y[1:], y_pred_arima),
            "RMSE": np.sqrt(mean_squared_error(y[1:], y_pred_arima)),
            "MAE": mean_absolute_error(y[1:], y_pred_arima),
            
        }
    
    if not np.isnan(y_pred_sarima).all():
        metrics["SARIMA"] = {
            "R² Train": "—", "R² Test": r2_score(y[13:], y_pred_sarima),
            "MSE": mean_squared_error(y[13:], y_pred_sarima),
            "RMSE": np.sqrt(mean_squared_error(y[13:], y_pred_sarima)),
            "MAE": mean_absolute_error(y[13:], y_pred_sarima),
            
        }
    
    # Déterminer le meilleur modèle
    metrics_clean = {k: v for k, v in metrics.items() if k != "VotingRegressor"}
    meilleur_modele = max(metrics_clean.items(), key=lambda x: x[1]["R² Test"] if isinstance(x[1]["R² Test"], (int, float)) else -np.inf)
    nom_meilleur_modele = meilleur_modele[0]
    score_meilleur = meilleur_modele[1]["R² Test"]


    # Arrondir R² Train à 4 chiffres significatifs
    for model in metrics:
        if isinstance(metrics[model]["R² Train"], (float, int)):
            metrics[model]["R² Train"] = float("{:.4g}".format(metrics[model]["R² Train"]))



    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Tableau des métriques
        st.subheader("Tableau des Métriques")
        df_metrics = pd.DataFrame(metrics).T
        df_metrics = df_metrics[["R² Train", "R² Test", "MSE", "RMSE", "MAE"]]
        df_metrics = df_metrics.round(3)
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Meilleurs hyperparamètres RandomForest")
        if 'grid_rf' in locals():
            params_rf = grid_rf.best_params_
            df_rf_params = pd.DataFrame(params_rf.items(), columns=["Paramètre", "Valeur"])
            st.table(df_rf_params)
            
            # Meilleur modèle sélectionné
            st.markdown("Meilleur modèle sélectionné")
            st.markdown(f"**{nom_meilleur_modele}** (R² Test = {score_meilleur:.3f})")
        else:
            st.write("Aucun paramètre trouvé pour RandomForest.")
    
    with col3:
        st.subheader("Meilleurs hyperparamètres XGBoost")
        if 'grid_xgb' in locals():
            params_xgb = grid_xgb.best_params_
            df_xgb_params = pd.DataFrame(params_xgb.items(), columns=["Paramètre", "Valeur"])
            st.table(df_xgb_params)
            
            
        else:
            st.write("Aucun paramètre trouvé pour XGBoost.")



    # Sélection de modèle pour affichage
    modele_visu = [k for k in metrics if k != "VotingRegressor"]
    choix_model = st.selectbox(
        "Choisissez un modèle pour la visualisation :",
        modele_visu,
        index=modele_visu.index(nom_meilleur_modele)
    )


    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_model["datetime"], y, label="Réel", color="blue")
    
    # Ajuster l’échelle de l’axe Y
    y_min, y_max = y.min(), y.max()
    margin = (y_max - y_min) * 0.1  # marge de 10%
    ax.set_ylim(y_min - margin, y_max + margin)


    if choix_model == "LinearRegression":
        ax.plot(df_model["datetime"], y_pred_lr, label="LR", color="orange")
    elif choix_model == "RandomForest":
        ax.plot(df_model["datetime"], y_pred_rf, label="RF", color="green")
    elif choix_model == "XGBoost":
        ax.plot(df_model["datetime"], y_pred_xgb, label="XGB", color="purple")
    elif choix_model == "ARIMA":
        ax.plot(df_model["datetime"].iloc[1:], y_pred_arima, label="ARIMA", color="red")
    elif choix_model == "SARIMA":
        ax.plot(df_model["datetime"].iloc[13:], y_pred_sarima, label="SARIMA", color="brown")

    ax.set_title(f"Prédiction avec modèle : {choix_model}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hauteur d'eau")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    
    
else:
    st.warning("Aucune donnée filtrée disponible pour l'entraînement des modèles.")



