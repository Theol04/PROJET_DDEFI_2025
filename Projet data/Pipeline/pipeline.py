import requests
import json
import pandas as pd
import numpy as np
import os

def get_data(url,params ):

  try:
      # Envoyer une requête GET
      response = requests.get(url,params)

      # Vérifier si la requête a réussi
      if response.status_code == 200:
          # Convertir la réponse JSON en objet Python
          data = response.json()
          print("Données récupérées avec succès :")
          #print(data)  # Affichez ou traitez les données ici
          results = data['results']
          df = pd.DataFrame(results)

          return df
      else:
          print(f"Erreur lors de la requête. Code HTTP : {response.status_code}")
          print("Message :", response.text)
          return pd.DataFrame(results)


  except requests.exceptions.RequestException as e:
      print("Une erreur est survenue :", e)
  return pd.DataFrame(results)

def download_data(url,departement,limit=9999999):
  """
  Permez de télécharger les data du departement choisi 
  limit permet de ne pas surcharger la mémoire et de pouvoir lancer le programme en plusieurs fois 
  """
  data_departement = pd.DataFrame()
  date = "2015-01-01"
  fichier_csv = os.path.join( "BDD", f"data_{departement}.csv")
  # Nom du fichier
  n= 0
  if os.path.exists(fichier_csv):
      data_departement = pd.read_csv(fichier_csv)  # Charger les données existantes
      date_debut = data_departement["date"].max()  # Continuer à partir de la dernière date récupérée
      print(f"Reprise à partir de la date : {date_debut}")
      date = date_debut

  while n<limit:
    n+=1
    try:
        params = {
            "select": "*",
            "where": f"nom_dept = '{departement}' AND date >= '{date}'",
            "order_by": "date",
            "limit": 100
        }

        data = get_data(url, params)
        if data.shape[0] <= 1:
            break

        data_departement = pd.concat([data_departement, data], ignore_index=True)
        data_departement = data_departement.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
        data_departement = data_departement.drop_duplicates()
        data_departement.to_csv(fichier_csv, index=False)
        date = data["date"].max()
    
    except Exception as e:
        print(f"Erreur : {e}")
  print(f"Date : {date} | Total récupéré : {data_departement.shape[0]}")



if __name__ == '__main__' : 
    url = "https://public.opendatasoft.com//api/explore/v2.1/catalog/datasets/donnees-synop-essentielles-omm/records"
    # Paramètres de la requête
    params = {
    "select": "AVG(rr24)",   # Sélectionne toutes les colonnes
    "group_by": "nom_dept",
    "limit": 200      # Limite à 20 enregistrements
}
    df = get_data(url,params)
    df_sorted = df.sort_values(by='AVG(rr24)', ascending=False)
    df_sorted