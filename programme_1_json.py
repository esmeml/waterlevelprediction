import os
import json

dossier = r"C:\Users\thiba\Git\projet_water\waterlevelprediction\data_France_1\data_France_1"

# Une seule liste pour contenir toutes les entrées
data_total = []

# Lecture de chaque fichier JSON
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith(".json"):
        chemin = os.path.join(dossier, nom_fichier)
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                contenu = json.load(f)
                if isinstance(contenu, list):
                    data_total.extend(contenu)  # ajoute plusieurs objets
                elif isinstance(contenu, dict):
                    data_total.append(contenu)  # ajoute un seul objet
        except Exception as e:
            print(f"Erreur dans {nom_fichier}: {e}")

# Structure unifiée pour pandas
output = {
    "data": data_total
}

# Sauvegarde dans le JSON
fichier_sortie = os.path.join(dossier, "dataset_fusionne.json")
with open(fichier_sortie, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"✅ Fusion réussie avec {len(data_total)} enregistrements dans : {fichier_sortie}")
