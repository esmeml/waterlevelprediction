import streamlit as st
import os
import json
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Observations Hydrologiques", layout="wide")
st.title("Information sur les cours d'eau")

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
    data = stations[choix]
    coords = data["geometry"]["coordinates"]
    initial_view = pdk.ViewState(latitude=coords[1], longitude=coords[0], zoom=8)
else:
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
