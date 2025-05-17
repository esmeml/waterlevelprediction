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

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.model_selection import TimeSeriesSplit ,  GridSearchCV


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



st.set_page_config(page_title="Observations Hydrologiques", layout="wide")
st.title("Information sur les cours d’eau")

DOSSIER_JSON = r"C:\Users\thiba\Git\projet_water\waterlevelprediction\projet_streamlit\data"

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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Informations")
        st.markdown(f"**Bassin** : {props.get('basin', 'N/A')}")
        st.markdown(f"**Pays** : {props.get('country', 'N/A')}")
        st.markdown(f"**Institution** : {props.get('institution', 'N/A')}")
        st.markdown(f"**Source** : {props.get('source', 'N/A')}")
        st.markdown(f"**Plateforme satellite** : {props.get('platform', 'N/A')}")
        st.markdown(f"**Statut** : {props.get('status', 'N/A')}")

    with col2:
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
                st.line_chart(df_filtré.set_index("datetime")["height"])
        else:
            st.warning("Aucune donnée de mesure trouvée.")


    with st.expander("Données brutes (table)"):
        if 'df_filtré' in locals() and not df_filtré.empty:
            st.dataframe(df_filtré[["datetime", "height", "associated_uncertainty", "satellite"]])
        else:
            st.write("Aucune donnée à afficher.")

    with st.expander("Métadonnées complètes"):
        st.json(props)



# Charger les résultats de modèle
MODELES_PATH = r"C:\Users\thiba\Git\projet_water\waterlevelprediction\projet_streamlit\resultats_modeles.json"
if os.path.exists(MODELES_PATH):
    with open(MODELES_PATH, "r", encoding="utf-8") as f:
        resultats_modeles = json.load(f)

    # Chercher un modèle qui correspond au cours d’eau sélectionné
    cle_modele_trouvee = None
    for key in resultats_modeles:
        nom_riviere = key.split("__")[0]
        noms_possibles = nom_riviere.split("_")
        if choix in noms_possibles:
            cle_modele_trouvee = key
            break

    if cle_modele_trouvee:
        modele_info = resultats_modeles[cle_modele_trouvee]
        st.subheader("Meilleur modèle prédictif")

        nom_modele = modele_info["best_model"]
        hyperparams = modele_info["model_params"]
        score = modele_info["best_score"]

        st.markdown(f"**Modèle :** `{nom_modele}`")
        st.markdown("**Meilleurs hyperparamètres :**")
        st.json(hyperparams)
        st.markdown(f"**Meilleur score :** {score}")
        st.subheader("Comparaison des valeurs réelles et prédites")

        if 'df_filtré' in locals() and not df_filtré.empty:
            lags = 5
            df_modele = df_filtré.copy()
            for i in range(1, lags + 1):
                df_modele[f"lag{i}"] = df_modele["height"].shift(i)
            df_modele.dropna(inplace=True)

            X = df_modele[[f"lag{i}" for i in range(1, lags + 1)]].copy()
            y = df_modele["height"].copy()
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Évaluer les scores de chaque modèle
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(**(hyperparams if nom_modele == "RandomForest" else {})),
                "XGBoost": XGBRegressor(**(hyperparams if nom_modele == "XGBoost" else {})),
                }

            scores = {}
            for name, mdl in models.items():
                mdl.fit(X_train, y_train)
                y_pred = mdl.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                scores[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}

        # Trouver le meilleur modèle selon le R²
            best_model_name_by_r2 = max(scores, key=lambda x: scores[x]["R2"])

            st.markdown("### Comparaison des performances des modèles")
            for name, metriques in scores.items():
                badge = "✨ **Meilleur**" if name == best_model_name_by_r2 else ""
                st.markdown(f"**{name}** {badge}")
                st.markdown(f"- R² : {metriques['R2']:.3f}")
                st.markdown(f"- RMSE : {metriques['RMSE']:.3f}")
                st.markdown(f"- MAE : {metriques['MAE']:.3f}")
                st.markdown("---")

    # Sélection dynamique du modèle
            default_model = nom_modele if nom_modele in scores else best_model_name_by_r2
            mode_selection1 = st.selectbox(
                "Choisissez un modèle",
                ["LinearRegression", "RandomForest", "XGBoost"],
                index=["LinearRegression", "RandomForest", "XGBoost"].index(default_model)
                )

    # Création du modèle sélectionné
            if mode_selection1 == "LinearRegression":
                model = LinearRegression()
            elif mode_selection1 == "RandomForest":
                model = RandomForestRegressor(**(hyperparams if nom_modele == "RandomForest" else {}))
            elif mode_selection1 == "XGBoost":
                model = XGBRegressor(**(hyperparams if nom_modele == "XGBoost" else {}))
            else:
                    st.error(f"Modèle non supporté : {mode_selection1}")
                    model = None

            if model is not None:
                model.fit(X_train, y_train)
                y_pred_historique = model.predict(X_test)
        
            # Dates correspondantes à y_test
                dates_test = df_modele.iloc[y_test.index]["datetime"]

                df_pred_historique = pd.DataFrame({
                    "datetime": dates_test.values,
                    "prediction": y_pred_historique
                    })  

        # Prédictions futures par prolongation des lags
                dernier_lag = X.iloc[-1].tolist()
                future_preds = []
                future_dates = []
                current_lag = dernier_lag.copy()
                last_date = df_modele["datetime"].max()

                for i in range(365):
                    input_array = np.array(current_lag).reshape(1, -1)
                    next_pred = model.predict(input_array)[0]
                    future_preds.append(next_pred)
                    next_date = last_date + pd.Timedelta(days=i + 1)
                    future_dates.append(next_date)

            # Mettre à jour les lags pour la prochaine prédiction
                    current_lag = current_lag[1:] + [next_pred]

                df_futur = pd.DataFrame({
                    "datetime": future_dates,
                    "prediction": future_preds
                    })

        # Affichage graphique
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_modele["datetime"], df_modele["height"], label='Valeurs réelles', color='blue')
                ax.plot(df_pred_historique["datetime"], df_pred_historique["prediction"], label='Valeurs prédites (test)', color='orange')
                ax.plot(df_futur["datetime"], df_futur["prediction"], label='Prédictions futures', linestyle='--', color='green')

                is_best = "✨" if mode_selection1 == best_model_name_by_r2 else ""
                ax.set_title(f"Prédictions avec modèle : {mode_selection1} {is_best}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Hauteur d'eau")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)
        else:
            st.warning("Aucune donnée filtrée disponible pour générer le graphique.")

    else:
        st.warning("Fichier des modèles non trouvé.")


