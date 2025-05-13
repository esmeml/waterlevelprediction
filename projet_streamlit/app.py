import streamlit as st
import os
import json
import pandas as pd

st.set_page_config(page_title="Observations Hydrologiques", layout="wide")
st.title("Information sur les cours d'eau")

DOSSIER_JSON = r"G:\projet_streamlit\data"

# Lister les fichiers JSON
fichiers = [f for f in os.listdir(DOSSIER_JSON) if f.endswith(".json")]

# Charger les données dans un dictionnaire
stations = {}
for fichier in fichiers:
    chemin = os.path.join(DOSSIER_JSON, fichier)
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
            river = data["properties"].get("river", fichier.replace(".json", ""))
            stations[river] = data
    except Exception as e:
        st.warning(f"Erreur lors du chargement de {fichier}: {e}")

# Sélectionner une station / fleuve
choix = st.selectbox("Sélectionnez un cours d’eau :", list(stations.keys()))

if choix:
    data = stations[choix]
    props = data["properties"]
    coords = data["geometry"]["coordinates"]
    mesures = data.get("data", [])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Position")
        st.map(pd.DataFrame([{"lat": coords[1], "lon": coords[0]}]))
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
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d %H:%M")
            df = df.sort_values("datetime")
            st.line_chart(df.set_index("datetime")["orthometric_height_of_water_surface_at_reference_position"])
            with st.expander("Données brutes (table)"):
                st.dataframe(df[["datetime", "orthometric_height_of_water_surface_at_reference_position", "associated_uncertainty", "satellite"]])
        else:
            st.warning("Aucune donnée de mesure trouvée.")

    with st.expander("Métadonnées complètes"):
        st.json(props)
