import streamlit as st
from flask import Flask
import requests
import json
import pickle
import pandas as pd
import numpy as np
import pandas as pd
#PyLab est un module pratique qui importe en bloc matplotlib.pyplot (pour le traçage) et NumPy (pour les mathématiques et l’utilisation de tableaux) #dans un seul espace de noms
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
plt.style.use('fivethirtyeight')
# Classifieur Xgboost
import xgboost
import shap
shap.initjs()
# Chargement et traitement d'image image
from PIL import Image

### Programme de traitement et d'affichage des données client ###
def main() :
    
    URL_API = "http://localhost:5000/"
    
    # Chargement des données
        #Lorsque nous marquons une fonction avec le décorateur de cache de Streamlit @st.cache, chaque fois que la fonction est appelée streamlit, vérifie            #les paramètres d’entrée avec lesquels vous avez appelé la fonction. Ce qui se passe dans le backend, c’est lorsqu’une fonction est décorée avec        
        #@st.cache streamlit conserve tous les états d’une fonction dans la mémoire.
    @st.cache()
    def chargement_compare_client():        
        with open('dossier_pkl/compare_client.pkl', 'rb') as f:                  
            compare_client =pickle.load(f)
        return compare_client 
     
       
    @st.cache()
    def chargement_compare_train():                
        with open('dossier_pkl/compare_train.pkl', 'rb') as f:                  
            compare_train =pickle.load(f)
        return compare_train 
      
    compare_train=chargement_compare_train()
     
        
    @st.cache()
    def chargement_test_shap():        
        with open('dossier_pkl/test_shap.pkl', 'rb') as f:                  
            test_shap =pickle.load(f)    
        return 
      
    Y=chargement_test_shap()
         
    @st.cache()
    def chargement_selectbox():
        # Requête permettant de récupérer la liste des ID clients
        data_json = requests.get(URL_API + "chargement_donnees")
        data =  data_json.json()        
        # Récupération des valeurs sans les [] de la réponse
        liste_id = []
        for i in data:
            liste_id.append(i)
        return liste_id
     
    
    @st.cache
    def chargement_informations_generales():
        # Requête permettant de récupérer :
        # Le nombre de lignes de crédits existants dans la base
        # Le revenus moyens des clients
        # Le montant moyen des crédits existants
        liste_informations = [compare_train.shape[0],
                round(compare_train["AMT_INCOME_TOTAL"].mean(), 2),
                round(compare_train["AMT_CREDIT"].mean(), 2)]
        nombre_credits =  liste_informations[0]
        moyenne_revenu =  liste_informations[1]
        moyenne_credits = liste_informations[2]

        # Requête permettant de récupérer
        # Le nombre de target dans la classe 0
        # et la classe 1
        target = requests.get(URL_API + "target")   
        # On transforme la réponse en dictionnaire python
        target  = json.loads(target.content.decode("utf-8"))
        target= pd.DataFrame(target)
        targets =  target.value_counts()
        return nombre_credits, moyenne_revenu, moyenne_credits, targets
    
    
    def identite_client(verification_identifiant):
        # Requête permettant de récupérer les informations du client sélectionné
        infos_client = requests.get(URL_API + "identite_client/", params={"id_client":verification_identifiant})
        # On transforme la réponse en dictionnaire python
        infos_client  = json.loads(infos_client.content.decode("utf-8"))
        # On transforme le dictionnaire en dataframe
        infos_client= pd.DataFrame.from_dict(infos_client).T

        return infos_client

    
    @st.cache
    def chargement_age_population():
        # Requête permettant de récupérer les âges de la 
        # population pour le graphique situant le client
        data_age_json = requests.get(URL_API + "chargement_age_population")
        data_age = data_age_json.json()
        return data_age

    @st.cache
    def chargement_revenu_population():
        # Requête permettant de récupérer des tranches de revenus 
        # de la population pour le graphique situant le client
        data_revenu_json = requests.get(URL_API + "chargement_revenu_population")
        data_revenu = data_revenu_json.json()
        # Récupération des valeurs sans les [] de la réponse
        liste_revenu = []
        for i in data_revenu:
            liste_revenu.append(i[0])
        return liste_revenu
        
    
    
    @st.cache
    def chargement_prediction_score(verification_identifiant):
       # Requête permettant de récupérer le score de la prédiction
        # de faillite du client sélectionné
        prediction_score_json = requests.get(URL_API + "chargement_prediction_score/", params={"id_client":verification_identifiant})
        Score_id = prediction_score_json.json()
        return Score_id
    
    @st.cache
    def chargement_prediction_label(verification_identifiant):
       # Requête permettant de récupérer le label de la prédiction
        # de faillite du client sélectionné
        prediction_label_json = requests.get(URL_API + "chargement_prediction_label/", params={"id_client":verification_identifiant})
        Label_id = prediction_label_json.json()
        return Label_id
    
    @st.cache
    def chargement_client_shap(verification_identifiant):
       # Requête permettant de récupérer les caractéristiques
        #  du client sélectionné pour l'affichage shappley
        client_shap_json = requests.get(URL_API + "chargement_client_shap/", params={"id_client":verification_identifiant})
         # On transforme la réponse en dictionnaire python
        client_shap  = json.loads(client_shap_json.content.decode("utf-8"))
        # On transforme le dictionnaire en dataframe
        client_shap= pd.DataFrame(client_shap, index=[0])
        return client_shap
    
     #######################################
    # SIDEBAR
    #######################################

    # Titre
    html_temp = """
    <div style="background-color: green; padding:10px; border-radius:10px">
    <h1 style="color: yellow; text-align:center">Tableau de bord de scoring crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision d'octroi du prêt …</p>
    """
    #st.write ecrit des arguments dans l’application
                #vous pouvez transmettre plusieurs arguments, qui seront tous écrits.
    #unsafe_allow_htm:Il s’agit d’un argument de mot-clé uniquement dont la valeur par défaut est False
                #Par défaut, toutes les balises HTML trouvées dans les chaînes seront traitées comme du texte pur. Ce comportement peut être désactivé en    
                 #définissant cet argument sur True.
    st.write(html_temp, unsafe_allow_html=True)
    
   ### Affichage logo Prêt à dépenser ###
    with open('dossier_pkl/im.pkl', 'rb') as f:                  
        im =pickle.load(f) 
    #st.sidebar: Cette fonction est utilisée pour créer une barre latérale sur le tableau de bord, puis nous pourrons mettre des données sur cette partie.
    #st.sidebar.columns: pour placer des colonnes dans la barre latérale, puis pour ajouter n’importe quel élément ou composant disponible à partir de la         bibliothèque Streamlit dans cette colonne
    colonne1, colonne2, colonne3 = st.sidebar.columns([30,250,60])
    #Les éléments peuvent être passés à st.sidebar à l’aide de la notation d’objet et de la notation with
    #Chaque élément transmis à st.sidebar est épinglé à gauche, ce qui permet aux utilisateurs de se concentrer sur le contenu de votre application.
    with colonne1:
        st.write("")
    with colonne2:
        st.image(im, width=250)
    with colonne3:
        st.write("")
    
    # Sélection identifiant client
    #header:en-tête
    st.sidebar.header("**Informations générales**")
   
    # Choix de l'identifiant
    #En utilisant la fonction selectbox(), nous allons créer une boîte de sélection des identifiants clients.
    identifiants_client =chargement_selectbox()
    global verification_identifiant
    verification_identifiant = st.sidebar.selectbox("Identifiant Client", identifiants_client)

    # Chargement des informations générales
    nombre_credits, moyenne_revenu, moyenne_credits, targets = chargement_informations_generales()


    ### Affichage des informations sur la sidebar ###
    # Nombre de prêts dans l'échantillon
    st.sidebar.write("<u>Nombre de prêts dans l'échantillon :</u>", unsafe_allow_html=True)
    st.sidebar.text(nombre_credits)

    # Moyenne des Revenus
    st.sidebar.write("<u>Moyenne des Revenus (EUROS) :</u>", unsafe_allow_html=True)
    st.sidebar.text(moyenne_revenu)

    # Moyenne des Montants de crédits
    st.sidebar.write("<u>Moyenne des Montants de crédits (EUROS) :</u>", unsafe_allow_html=True)
    st.sidebar.text(moyenne_credits)
    
    #Distribution: PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Non défaillant', 'Défaillant'], autopct='%1.1f%%', startangle=90)
    #st.pyplot () : cette fonction permet d’afficher une figure matplotlib.pyplot
    st.sidebar.pyplot(fig)
    
    
    
    
    #######################################
    # CONTENU DE LA PAGE PRINCIPALE
    #######################################
    # Affichage de l'identifiant client depuis la Sidebar
    st.write("Sélection identifiant client :", verification_identifiant)


    # Affichage des informations client : Sexe, Age, Status familial, Enfants, …
    st.header("**Informations du client**")

    affiche_client = 0
    
    if st.checkbox("Informations relatives au client.", value=True):
        
        affiche_client = 1
        global infos_client
        infos_client = identite_client(verification_identifiant)
        
        st.write("**Sexe : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["AGE"].values[0])))
        st.write("**Status familial : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
        
        # Affecter 'WEIGHTED_EXT_SOURCE' à une variable pour un pie chart comparatif en fin de dashboard
        w_score_client = round(infos_client['WEIGHTED_EXT_SOURCE'].values[0], 2)
    
        # Histogramme des âges
        data_age = chargement_age_population()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="blue", bins=20)
        ax.axvline(int(infos_client["AGE"].values), color="green", linestyle='--')
        ax.set(title="Position du client dans l'histogramme des âges", xlabel='Age(Années)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Revenu (EUROS)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Ratio-Annuités de crédit : **{:.0f}".format(infos_client["CREDIT_ANNUITY_RATIO"].values[0]))
        st.write("**Valeur du bien financé : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
        
        # Histogramme des revenus
        data_revenu = chargement_revenu_population()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_revenu, edgecolor = 'k', color="blue", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title="Position du client dans l'histogramme des revenus", xlabel='Revenu (EUROS)', ylabel='')
        st.pyplot(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        st.write("Veuillez préalablement afficher les informations relatives au client !")

    # Solvabilité du client
    st.header("**Analyse du dossier client**")
    Score = chargement_prediction_score(verification_identifiant)
    Label = chargement_prediction_label(verification_identifiant)
    st.write(verification_identifiant)
    if Label == 1:
        st.write("**Défaillant avec une probabilité de : **{:.0f} %".format(round(float(Score)*100, 2)))
    else:
        st.write("**Non Défaillant avec une probabilité de : **{:.0f} %".format(round(float(Score)*100, 2)))
        
    # Dataframe avec l'ensemble des caractérisques du client
    colonnes_a_afficher = ['CODE_GENDER', 'AGE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT', 'CREDIT_ANNUITY_RATIO','NAME_CONTRACT_TYPE','AMT_GOODS_PRICE', 'WEIGHTED_EXT_SOURCE']
    st.markdown("<u>Données du client :</u>", unsafe_allow_html=True)
    st.write(infos_client[colonnes_a_afficher])

    # Importance Fonctionnalité / SHAP Values
    
    if st.checkbox("Identifiant client {:.0f} : caractéristiques importantes.".format(verification_identifiant), value=True):
        shap.initjs()
        # Données shappley_values  pour l'affichage shappley 
        with open('dossier_pkl/shap_values.pkl', 'rb') as f:    
            shap_values=pickle.load(f) 
    
        client_shap = chargement_client_shap(verification_identifiant)
            
        # Interprétation et Affichage du bar plot des features importances
        fig, ax = plt.subplots(figsize=(10, 10))
        #afficher un widget de curseur
        number = st.slider("Choix du nombre de caratéristiques du client …", 0, 20, 8)
        shap.summary_plot(shap_values, client_shap, max_display=number, plot_type ="bar", color_bar=False)
        st.pyplot(fig)
        
    else:
        st.write("<i>…</i>", unsafe_allow_html=True)
        
    
    # Afficahe des principales caractéristiques des clients similaires défaillants et non défaillants
    
    if st.checkbox("Prinicipales caractéristiques de clients similaires selon les critères de : sexe, status familial, âge, revenu, montant du crédit."):
        compare_client = chargement_compare_client()
        if affiche_client == 1:
        
            # Masques de sélection
            sexe = infos_client['CODE_GENDER'].values[0]
            age = infos_client['AGE'].values[0]
            revenu = infos_client['AMT_INCOME_TOTAL'].values[0]
            credit = infos_client['AMT_CREDIT'].values[0]
            status = infos_client['NAME_FAMILY_STATUS'].values[0]
            child = infos_client['CNT_CHILDREN'].values[0]

            mask_1 = compare_client['CODE_GENDER'] == sexe
            mask_2 = compare_client['NAME_FAMILY_STATUS'] == status
            mask_3 = (compare_client['DAYS_BIRTH'] > 0.80 * age) & (compare_client['DAYS_BIRTH'] < 1.20 * age)
            mask_4 = (compare_client['AMT_INCOME_TOTAL'] > 0.50 * revenu) & (compare_client['AMT_INCOME_TOTAL'] < 1.5 * revenu)
            mask_5 = (compare_client['AMT_CREDIT'] > 0.50 * credit) & (compare_client['AMT_CREDIT'] < 1.50 * credit)

            # Clients avec un profil similaire défaillants
            st.write("**Clients avec un profil similaire défaillant**")
            mask_0 = compare_client['TARGET'] == 1
            data_compare = compare_client[mask_0 & mask_1 & mask_2 & mask_3 & mask_4 & mask_5 ]
            data_compare = data_compare[[ 'SK_ID_CURR','CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', \
                                                'AMT_CREDIT', 'CREDIT_ANNUITY_RATIO', 'NAME_CONTRACT_TYPE','AMT_GOODS_PRICE', 'WEIGHTED_EXT_SOURCE', 'TARGET']]
            data_compare = data_compare.rename({'DAYS_BIRTH':'AGE'}, axis=1)
            data_compare['AGE'] =  data_compare['AGE'].astype(int)

            # Affecter 'WEIGHTED_EXT_SOURCE' à une variable pour pie chart comparatif en fin de dashboard
            w_score_defaillant = round(data_compare['WEIGHTED_EXT_SOURCE'].mean(), 2)
            
            if np.math.isnan(w_score_defaillant):
                st.write('Selon les critères de similarité retenus, pas de clients comparables à afficher.')
            else:
                st.write(data_compare)

            # Clients avec un profil similaire non défaillants
            st.write("**Clients avec un profil similaire non défaillant**")
            mask_0 = compare_client['TARGET'] == 0
            data_compare = compare_client[mask_0 & mask_1 & mask_2 & mask_3 & mask_4 & mask_5 ]
            data_compare = data_compare[[ 'SK_ID_CURR','CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', \
                                                'AMT_CREDIT', 'CREDIT_ANNUITY_RATIO', 'NAME_CONTRACT_TYPE','AMT_GOODS_PRICE', 'WEIGHTED_EXT_SOURCE', 'TARGET']]
            data_compare = data_compare.rename({'DAYS_BIRTH':'AGE'}, axis=1)
            data_compare['AGE'] = data_compare['AGE'].astype(int)

            # Affecter 'WEIGHTED_EXT_SOURCE' à une avariable pour pie chart comparatif en fin de dashboard
            w_score_non_defaillant = round(data_compare['WEIGHTED_EXT_SOURCE'].mean(), 2)
            
            if np.math.isnan(w_score_non_defaillant):
                st.write('Selon les critères de similarité retenus, pas de clients comparables à afficher.')
            else:
                st.write(data_compare)

            # Afficher le pie chart des scores normalisés comparés entre le client, les défaillants et les non défaillants 
            st.write("**Score normalisé comparatif **")

            fig, ax = plt.subplots(figsize=(1,1))
            scores_normalises = [w_score_client, w_score_defaillant,w_score_non_defaillant]
            plt.pie(scores_normalises, labels=['Client', 'Défaillant', 'Non défaillant'], autopct='%1.1f%%', textprops={'fontsize': 5}, startangle=90)
            st.pyplot(fig)

            st.write("Score normalisé du client : ", w_score_client)
            st.write("Score normalisé moyen des défaillants : ", w_score_defaillant)
            st.write("Score normalisé moyen des non défaillants : ", w_score_non_defaillant)
            
    else:        
        st.write("Veuillez préalablement afficher les prinicipales caractéristiques de clients similaires au client !")
        
    st.markdown('***')
    st.markdown("**Outil d'aide à la décision développé par la groupe Prêt à dépenser**.")


if __name__ == '__main__':
    main()
        