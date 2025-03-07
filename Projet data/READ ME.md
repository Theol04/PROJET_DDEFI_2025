*PREDICTION DES PRECIPITATIONS A 24H*

**Description**

Ce projet vise à prédire les précipitations dans les 24 prochaines heures pour un département donné. Il repose sur l'utilisation de données météorologiques accessibles en ligne et l'entraînement d'un modèle de machine learning pour effectuer des prédictions précises.

**Architecture**

Ce projet est organisé en plusieurs fichiers, chacun ayant un rôle spécifique :

    1️ - pipeline.py - Récupération des données météorologiques

Ce fichier est responsable de la collecte des données météorologiques à partir d'une API publique. Il contient les fonctions suivantes :

get_data(url, params) : Envoie une requête GET à l'API et récupère les données météorologiques sous forme de DataFrame.

download_data(url, departement, limit=9999999) : Récupère les données météorologiques pour un département donné, en s'assurant que les données ne sont pas dupliquées. Les données sont stockées dans un fichier CSV sous BDD/data_<departement>.csv.

_Exemple d'utilisation_ :

```from pipeline import download_data

departement = "Finistère"
url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/donnees-synop-essentielles-omm/records"
download_data(url, departement)```


    2️ -  ml.py - Entraînement du modèle de machine learning

Ce fichier est dédié à l'entraînement d'un modèle de machine learning pour la prédiction des précipitations. Principales fonctions :

load_data(departement) : Charge les données météorologiques d'un département spécifique.

target_data(df, features) : Prépare les données pour l'entraînement en sélectionnant les colonnes d'intérêt et en définissant la variable cible rr24_future.

save_model(model, folder_path, metrics, features, grid_param, metric_to_compare, lower_is_better) : Sauvegarde le meilleur modèle en fonction d'une métrique de performance.

grid_search_model(X_train, y_train) : Utilise GridSearchCV pour optimiser un modèle RandomForestRegressor.

ml(departement, features_to_keep, accuracy_tolerance, metric_to_compare='Accuracy', lower_is_better=False) : Entraîne un modèle avec les paramètres définis et sauvegarde le meilleur modèle.

load_best_model(departement) : Charge le modèle le plus performant pour un département donné.

_Exemple d'entraînement du modèle_ :

```from ml import ml

departement = "Finistère"
features = ["t", "u", "pmer", "ff", "vv"]  # Variables utilisées
ml(departement, features, accuracy_tolerance=0.2)```


    3️ - main.py - Prédiction des précipitations

Ce fichier utilise les données météorologiques et le modèle de machine learning pour effectuer des prédictions. Fonction principale :

predict_rain_in_24h(departement) :

    Télécharge et met à jour les données météorologiques du département.

    Charge le modèle pré-entraîné et récupère les features utilisées.

    Vérifie que toutes les colonnes nécessaires sont présentes.

    Réalise une prédiction de la quantité de pluie prévue dans 24h.

_Exemple d'utilisation_ :

```from main import predict_rain_in_24h

departement = "Finistère"
prediction = predict_rain_in_24h(departement)
print(f"Précipitations prévues dans 24h : {prediction:.2f} mm")```


***GitHub : https://github.com/Theol04/PROJET_DDEFI_2025