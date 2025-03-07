import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os 
import pickle
import json 
import numpy as np 
from sklearn.metrics import make_scorer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Pipeline.pipeline import download_data

class Config:
    TOLERANCE = 0.1 

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Crée le dossier
        print(f"Dossier créé : {folder_path}")
    else:
        print(f"Le dossier existe déjà : {folder_path}")
    
def load_data(departement): 
        # Charger les données
    url = "https://public.opendatasoft.com//api/explore/v2.1/catalog/datasets/donnees-synop-essentielles-omm/records"

    download_data(url,departement) 
    fichier_csv = os.path.join( "BDD", f"data_{departement}.csv")  
      
    #Creer le fichier ou va être stocké le meilleur modèle ML 
    if __name__ == '__main__' : 
        folder_path = os.path.join("best_ml_by_dept", departement)  # Crée un chemin relatif au script
    else : 
        folder_path = os.path.join('ML',"best_ml_by_dept", departement)
    create_folder(folder_path)
    
    df = pd.read_csv(fichier_csv, parse_dates=["date"],low_memory=False)
    return df, folder_path

def target_data(df,features): 
    df = df.sort_values("date")  # Trier les données par date

    # Vérifier que 'rr24' existe dans df rr24
    if "rr24" not in df.columns:
        raise ValueError("La colonne 'rr24' est absente des données !")

    # Décaler rr24 pour en faire la cible future
    df["rr24_future"] = df["rr24"].shift(-8)  # Décalage d'une ligne vers le haut

    # Vérifier que toutes les colonnes features existent
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Supprimer les lignes avec NaN après le shift
    df = df.dropna(subset=["rr24_future"])  # Supprime seulement si `rr24_future` est NaN

    # Séparer les features (X) et la cible (y)
    X = df[features]
    y = df["rr24_future"]

    print(f"✅ Taille de X : {X.shape}, Taille de y : {y.shape}")
    
    return X, y


def save_model(model, folder_path, metrics, features,grid_param ,metric_to_compare="MAE", lower_is_better=True):
    """
    Enregistre un modèle et ses métriques uniquement s'il est meilleur que l'ancien modèle.

    Args:
        model (sklearn model): Modèle entraîné à sauvegarder.
        folder_path (str): Chemin du dossier où enregistrer le modèle.
        metrics (dict): Dictionnaire contenant les métriques du modèle.
        features (list): Liste des features utilisées pour entraîner le modèle.
        metric_to_compare (str): Nom de la métrique à comparer (ex: "MAE", "R2").
        lower_is_better (bool): Si True, une valeur plus basse de la métrique est meilleure (ex: MAE, RMSE).
                                Si False, une valeur plus haute est meilleure (ex: R2).

    Returns:
        bool: True si le modèle a été enregistré, False sinon.
    """

    # Création du dossier s'il n'existe pas
    os.makedirs(folder_path, exist_ok=True)

    # Définition des chemins des fichiers
    model_path = os.path.join(folder_path, "best_model.pkl")
    metrics_path = os.path.join(folder_path, "metrics.json")
    features_path = os.path.join(folder_path, "features.json")
    grid_param_path = os.path.join(folder_path, "grid_param.json")

    # Vérifier si un ancien modèle existe et charger ses métriques
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            old_metrics = json.load(f)
        old_value = old_metrics.get(metric_to_compare, None)  # Récupérer la métrique à comparer
    else:
        old_value = None  # Aucun ancien modèle existant

    new_value = metrics[metric_to_compare]

    # Vérification : enregistrer si meilleur ou si aucun ancien modèle
    should_save = False
    if old_value is None:
        should_save = True  # Pas d'ancien modèle => enregistrer
    else:
        if lower_is_better:
            should_save = new_value < old_value  # Plus bas = meilleur
        else:
            should_save = new_value > old_value  # Plus haut = meilleur

    # Sauvegarde uniquement si le modèle est meilleur
    if should_save:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(features_path, "w") as f:
            json.dump({"features": features}, f, indent=4)  
        
        with open(grid_param_path, "w") as f:
            json.dump(grid_param, f, indent=4)

        print(f"✅ Nouveau modèle enregistré dans {folder_path} avec {metric_to_compare}: {new_value}")
        print(f"📌 Features utilisées : {features}")
        return True
    else:
        print(f"❌ Modèle NON enregistré, ancien modèle était meilleur :({metric_to_compare}: {old_value}) nouveau modèle : {metric_to_compare}: {new_value}")
        return False

import numpy as np

def accuracy_with_tolerance(y_test, y_pred):
    """
    Calcul le pourcentage de grosse pluie bien annncé 
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    correct_predictions = ((y_pred >= Config.TOLERANCE) & (y_test >= Config.TOLERANCE))| ((y_pred < Config.TOLERANCE) & (y_test < Config.TOLERANCE))

    return round(np.mean(correct_predictions) * 100, 2)

def accuracy_with_tolerance_scorer(y_true, y_pred):
    """
    Wrapper pour utiliser accuracy_with_tolerance dans GridSearchCV.
    """
    return accuracy_with_tolerance(y_true, y_pred)

# Créer un scorer compatible avec GridSearchCV (plus haut = meilleur)


def grid_search_model(X_train, y_train):
    """
    Utilise GridSearchCV pour optimiser RandomForestRegressor avec accuracy_with_tolerance.
    """
    accuracy_scorer = make_scorer(accuracy_with_tolerance_scorer, greater_is_better=True)
    model,param_grid = model_selected()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=accuracy_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"✅ Meilleurs paramètres (basé sur l'accuracy +{Config.TOLERANCE}mm de pluie) : {grid_search.best_params_}")

    return grid_search.best_estimator_ , grid_search.best_params_

def ml(departement,features_to_keep,accuracy_tolerance, metric_to_compare='Accuracy',lower_is_better=False) : 
    
    #charge le data frame et le dossier où enregistrer le modèle 
    df,folder_path = load_data(departement)

    #selection des des features pour entrainer le modèle 
    Config.TOLERANCE = accuracy_tolerance

    #X : data_features ; y: data_target 
    X,y = target_data(df, features_to_keep)
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Entraîner un modèle Random Forest
    model,grid_param = grid_search_model(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)


    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "MSE": mean_squared_error(y_test, y_pred), 
        "RMSE": round(mean_squared_error(y_test, y_pred) ** 0.5, 2),
        "R2": round(r2_score(y_test, y_pred), 4),
        "Accuracy": accuracy_with_tolerance(y_test,y_pred)
    }
        
    save_model(model, folder_path, metrics, features_to_keep,grid_param, metric_to_compare, lower_is_better)

def load_best_model(departement):
    """
    Charge le meilleur modèle ML pour un département donné.

    Args:
        departement (str): Nom du département.

    Returns:
        model (sklearn model): Modèle chargé.
        features (list): Liste des features utilisées pour l'entraînement.
    """
    folder_path = os.path.join("best_ml_by_dept", departement)

    model_path = os.path.join(folder_path, "best_model.pkl")
    features_path = os.path.join(folder_path, "features.json")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"❌ Aucun modèle trouvé pour {departement}. Entraînement du modèle !")

        # Charger le modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Charger les features utilisées lors de l'entraînement
    with open(features_path, "r") as f:
        features = json.load(f)["features"]

    print(f"✅ Modèle chargé pour {departement} avec les features : {features}")

    return model, features

def model_selected(): 
    model =RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
    return model, param_grid

if __name__ == '__main__' : 
    

    
    features_to_keep = ["mois_de_l_annee","t", "td", "u", "pmer", "pres", "ff", "raf10", "n", "vv"]
    departement = "Bas-Rhin"
    ml(departement,features_to_keep,accuracy_tolerance= 5)