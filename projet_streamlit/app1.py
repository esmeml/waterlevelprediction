import streamlit as st
import os
import json
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


st.set_page_config(page_title="Observations Hydrologiques", layout="wide")
st.title("Information sur les cours d’eau")

DOSSIER_JSON = r"C:\Users\niels\Git\waterlevelprediction\projet_streamlit\data"

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
MODELES_PATH = r"C:\Users\niels\Git\waterlevelprediction\projet_streamlit\resultats_modeles.json"
if os.path.exists(MODELES_PATH):
    with open(MODELES_PATH, "r", encoding="utf-8") as f:
        resultats_modeles = json.load(f)

    # Chercher un modèle qui correspond au cours d’eau sélectionné
    cle_modele_trouvee = None
    for key in resultats_modeles:
        # Exemple de clé : "Adour_Adour__2016-08-06__2025-03-04.json"
        nom_riviere = key.split("__")[0]  # "Adour_Adour"
        noms_possibles = nom_riviere.split("_")  # ["Adour", "Adour"]
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

        # Recréer le DataFrame filtré sur lequel on va entraîner/prédire
        if 'df_filtré' in locals() and not df_filtré.empty:
            # Préparation des données
            df_modele = df_filtré.copy()
            df_modele = df_modele.dropna(subset=["height"])  # Supprimer les NaNs
            df_modele["timestamp"] = df_modele["datetime"].astype(np.int64) // 10**9  # Secondes depuis epoch

            X = df_modele[["timestamp"]]
            y = df_modele["height"]

            # Création du modèle selon le nom
            if nom_modele == "LinearRegression":
                model = LinearRegression(**hyperparams)
            elif nom_modele == "RandomForest":
                model = RandomForestRegressor(**hyperparams)
            elif nom_modele == "XGBoost":
                model = XGBRegressor(**hyperparams)
            else:
                st.error(f"Modèle non supporté : {nom_modele}")
                model = None

            # Entraînement et prédiction
            if model:
                model.fit(X, y)
                df_modele["prediction"] = model.predict(X)

                # Affichage de la courbe réelle vs prédite
                st.subheader("Comparaison : données réelles vs prédictions du modèle")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_modele["datetime"], df_modele["height"], label="Observé", color="blue")
                ax.plot(df_modele["datetime"], df_modele["prediction"], label="Prédit", color="red", linestyle="--")
                ax.set_xlabel("Date")
                ax.set_ylabel("Hauteur d'eau")
                ax.set_title(f"Modèle : {nom_modele}")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Aucune donnée filtrée disponible pour générer le graphique.")
    
else:
    st.warning("Fichier des modèles non trouvé.")


