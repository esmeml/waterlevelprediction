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
    # Vue centrée par défaut sur les points entre 40° et 50°N
    df_filtre = df_coords[(df_coords["lat"] >= 40) & (df_coords["lat"] <= 50)]

    if not df_filtre.empty:
        initial_view = pdk.ViewState(
            latitude=df_filtre["lat"].mean(),
            longitude=df_filtre["lon"].mean(),
            zoom=4
            )
        
    else:
        initial_view = pdk.ViewState(latitude=0, longitude=0, zoom=1)

# Affichage unique de la carte avec tous les points visibles
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
            st.line_chart(df.set_index("datetime")["height"])
            
        else:
            st.warning("Aucune donnée de mesure trouvée.")
    with st.expander("Données brutes (table)"):
        st.dataframe(df[["datetime", "height", "associated_uncertainty", "satellite"]])
    with st.expander("Métadonnées complètes"):
        st.json(props)
