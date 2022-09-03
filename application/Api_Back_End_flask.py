from flask import Flask, jsonify, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np

# Classifieur Xgboost
import xgboost
# Librairie Pycaret et scikit-learn
import pycaret
#from pycaret.classification import *
from pycaret.utils import check_metric
from sklearn.metrics import log_loss
from pycaret.classification import load_model, predict_model
from sklearn.model_selection import train_test_split

# Créez une instance de la classe Flask qui est l’application WSGI.
# Le premier argument est le nom du module ou du package d’application,
# généralement __name__ lors de l’utilisation d’un seul module.
app = Flask(__name__)


#Chargement des données
# Informations sur le client choisi dans le jeu de données Test sans Target
with open('dossier_pkl/informations_client_test.pkl', 'rb') as f:                  
    informations_client_test =pickle.load(f)
with open('dossier_pkl/selection_clients.pkl', 'rb') as f:                  
    selection_clients =pickle.load(f)

# Jeu de données pour les comparaisons dans la jeu de données Train avec Target
with open('dossier_pkl/compare_train.pkl', 'rb') as f:                  
    compare_train =pickle.load(f)
with open('dossier_pkl/compare_client.pkl', 'rb') as f:                  
    compare_client =pickle.load(f)

# Jeu de données pour la prédiction dans le jeu de données Test avec le modèle du classifieur final Xgboost 
with open('dossier_pkl/data_test_std_300_sample.pkl', 'rb') as f:                  
    data_test_std_300_sample =pickle.load(f)

# Jeux de données pour l'importance des fonctionnalités(SHAP Values)
with open('dossier_pkl/train_shap.pkl', 'rb') as f:                  
    train_shap =pickle.load(f) 
with open('dossier_pkl/test_shap.pkl', 'rb') as f:                  
    test_shap =pickle.load(f) 
with open('dossier_pkl/y_shap.pkl', 'rb') as f:                  
    y_shap=pickle.load(f) 
    
# Modèle retenue xgboost pour la prédiction
with open('dossier_pkl/clf.pkl', 'rb') as f:    
    clf =pickle.load(f) 
    prediction_test = predict_model(clf,  probability_threshold = 0.74, data = data_test_std_300_sample)
    
    
# Renommer la variable 'DAYS_BIRTH' en  'AGE' et la convertir en integer
informations_client_test =  informations_client_test.rename({'DAYS_BIRTH':'AGE'}, axis=1)
informations_client_test['AGE'] =  informations_client_test['AGE'].astype(int)


# Routage des décorateurs par Flask.
# Pour ajouter d’autres ressources, créez des fonctions qui génèrent le contenu de la page
# et ajoutez des décorateurs pour définir les localisateurs de ressources appropriés pour eux.


# Chargement des données compare_client
@app.route("/chargement_compare_client", methods=["GET"])
def chargement_compare_client():    
    return  compare_client.to_json(orient='values')

# Chargement des données compare_train
@app.route("/chargement_compare_train", methods=["GET"])
def chargement_compare_train():    
    return  compare_train.to_json(orient='values')

# Chargement des données pour la selection de l'ID client
@app.route("/chargement_donnees", methods=["GET"])
def chargement_donnees():    
    return  jsonify(selection_clients)

# Chargement d'informations générales
@app.route("/chargement_informations_generales", methods=["GET"])
def chargement_informations_generales():
    liste_informations = [compare_train.shape[0],
                 round(compare_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(compare_train["AMT_CREDIT"].mean(), 2)]
    
    return jsonify(liste_informations)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/target", methods=["GET"])
def target():
    target = compare_train.TARGET
    return target.to_json(orient="values")

@app.route("/identite_client/", methods=["GET"])
def identite_client():
    id = request.args.get("id_client")
    data_client =  informations_client_test[informations_client_test.index == int(id)]
    response = json.loads(data_client.to_json(orient='index'))
    return response

@app.route("/chargement_age_population", methods=["GET"])
def chargement_age_population():
    data_age = round((informations_client_test["AGE"]), 2)
    return data_age.to_json(orient='values')

@app.route("/chargement_revenu_population", methods=["GET"])
def chargement_revenu_population():
    data_revenu = informations_client_test[["AMT_INCOME_TOTAL"]]
    data_revenu = data_revenu[data_revenu['AMT_INCOME_TOTAL'] < 200000]
    return data_revenu.to_json(orient='values')

@app.route("/chargement_prediction_score/", methods=["GET"])
def chargement_prediction_score():
    id = request.args.get("id_client")
    Score = float(prediction_test[prediction_test['SK_ID_CURR'] == int(id)].Score.values)   
    return jsonify(Score)

@app.route("/chargement_prediction_label/", methods=["GET"])
def chargement_prediction_label():
    id = request.args.get("id_client")
    Label = int(prediction_test[prediction_test['SK_ID_CURR'] == int(id)].Label.values)
    return jsonify(Label)


@app.route("/chargement_prediction/", methods=["GET"])
def chargement_prediction():
    id = request.args.get("id_client")
    Score = float(prediction_test[prediction_test['SK_ID_CURR'] == int(id)].Score.values)   
    Label = int(prediction_test[prediction_test['SK_ID_CURR'] == int(id)].Label.values)
    if Label == 1:
        return "**Défaillant avec une probabilité de : **{:.0f} %".format(round(float(Score)*100, 2))
    else:
        return "**Non Défaillant avec une probabilité de : **{:.0f} %".format(round(float(Score)*100, 2))
    

@app.route("/chargement_client_shap/", methods=["GET"])
def chargement_client_shap():
    id = request.args.get("id_client")
    client_shap = test_shap.loc[int(id), : ]   
    return client_shap.to_json(orient='columns')


#if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)    
if __name__ == "__main__":
    app.run(host="0.0.0.0")
    #app.run(host="localhost", port="5000", debug=True, use_reloader=False)