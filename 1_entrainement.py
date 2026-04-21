#  Analyse de sentiments IMDB — Pipeline complet KNN
#  Dépendances : pip install pandas scikit-learn nltk beautifulsoup4

import re
import pandas as pd
import scipy.sparse as sp
 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk

# 1. prétraitement : nettoyer les données
"""
Il est noté qu'il existe des <br /> étiquettes
"""
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def pretraitement(text):
    # retirer les html étiquettes par la librairie BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # en miniscule
    text = text.lower()
    
    # supprimer les ponctuations et les symboles spéciales
    text = re.sub(r'[^a-z\s]', '', text)
    
    # enlever les stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

# 2. chargement des corpus
train_df = pd.read_csv('./dataset/imdb_train.csv')
val_df   = pd.read_csv('./dataset/imdb_validation.csv')
test_df  = pd.read_csv('./dataset/imdb_test.csv')

print(f"Train : {len(train_df)} | Validation : {len(val_df)} | Test : {len(test_df)}")

print("Terminé en charge des données d'entrainement.")

print("Prétraitement en cours...")
X_train_texte = train_df['review'].apply(pretraitement)
X_val_texte   = val_df['review'].apply(pretraitement)
X_test_texte  = test_df['review'].apply(pretraitement)

print("Terminé en prétraitement de corpus.")
      
# 3. vectorisation TF-IDF
vectoriseur = TfidfVectorizer(max_features=10000)
X_train = vectoriseur.fit_transform(X_train_texte)  # fit uniquement sur train
X_val   = vectoriseur.transform(X_val_texte)
X_test  = vectoriseur.transform(X_test_texte)

# 4. Sauvegarde des matrices et labels 
sp.save_npz('X_train.npz', X_train)
sp.save_npz('X_val.npz',   X_val)
sp.save_npz('X_test.npz',  X_test)
 
train_df['sentiment'].to_csv('y_train.csv', index=False)
val_df['sentiment'].to_csv('y_val.csv',     index=False)
test_df['sentiment'].to_csv('y_test.csv',   index=False)
 
print("Fichiers sauvegardés : X_train.npz, X_val.npz, X_test.npz, y_*.csv")
print("Veuillez lancer maintenant 2_validation.py")