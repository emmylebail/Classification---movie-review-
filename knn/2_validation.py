#  Script 2 — Validation : sélection du meilleur K
#  Nécessite : X_train.npz, X_val.npz, y_train.csv, y_val.csv
#  Génère    : meilleur_k.txt, knn_selection_k.png
 
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# chargement des corpus train et validation
X_train = sp.load_npz('X_train.npz')
X_val   = sp.load_npz('X_val.npz')
y_train = pd.read_csv('y_train.csv')['sentiment']
y_val   = pd.read_csv('y_val.csv')['sentiment']

# Test de plusieurs valeurs de K
"""K_VALUES = [1, 5, 10, 50, 100, 200, 400, 600, 800, 1000]
K_VALUES = [350, 370, 390, 400, 410, 430, 450, 500, 550, 600] 
K_VALUES = [240, 260, 280, 300, 320, 330, 340, 350, 360] 
K_VALUES = [335, 340, 342, 345, 348, 350] # après quelques suites de petits tests de dichonomy avec ces k, on trouve le meilleur k 
"""

K_VALUES = [335, 340, 342, 345, 348, 350]
scores = {}

for k in K_VALUES:
    print(f"Test K={k} ...", end=' ', flush=True)
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average='weighted')
    
    scores[k] = f1  # pour choisir le meilleur k : F1 weighted
    print(f"Accuracy = {acc:.4f} | F1 (weighted) = {f1:.4f}")

meilleur_k = max(scores, key=scores.get)
print(f"\nMeilleur K = {meilleur_k}  (F1 weighted = {scores[meilleur_k]:.4f})")

with open('meilleur_k.txt', 'w') as f:
    f.write(str(meilleur_k)) # sauvegarder le meilleur k afin de faciliter l'évaluation suivante

plt.figure(figsize=(6, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker='o', color='steelblue', linewidth=2)
plt.axvline(meilleur_k, color='tomato', linestyle='--', label=f'Meilleur K={meilleur_k}')
plt.xlabel('Valeur de K')
plt.ylabel('F1 weighted (validation)')
plt.title('Sélection du K optimal')
plt.legend()
plt.tight_layout()
plt.savefig('knn_selection_k.png', dpi=150)
plt.close()
 
print("Fichiers sauvegardés : meilleur_k.txt, knn_selection_k.png")
print("Veuillez lancer maintenant : python 3_evaluation.py")