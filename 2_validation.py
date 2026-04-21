#  Script 2 — Validation : sélection du meilleur K
#  Nécessite : X_train.npz, X_val.npz, y_train.csv, y_val.csv
#  Génère    : meilleur_k.txt, knn_selection_k.png
 
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# chargement des corpus train et validation
X_train = sp.load_npz('X_train.npz')
X_val   = sp.load_npz('X_val.npz')
y_train = pd.read_csv('y_train.csv')['sentiment']
y_val   = pd.read_csv('y_val.csv')['sentiment']

# Test de plusieurs valeurs de K
"""K_VALUES = [1, 15, 47,89, 151, 551, 661, 1551]
K_VALUES = [401, 451, 501, 551, 601, 651, 701] 
K_VALUES = [561, 571, 581, 591, 601, 611, 621] # après quelques suites de petits tests de dichonomy avec ces k, on trouve le meilleur k 
"""

K_VALUES = [400, 591, 601, 603, 609, 611, 613, 621, 701]
accuracies = {}
 
for k in K_VALUES:
    print(f"Test K={k} ...", end=' ', flush=True)
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    accuracies[k] = acc
    print(f"Accuracy = {acc:.4f}")
 
meilleur_k = max(accuracies, key=accuracies.get)
print(f"\nMeilleur K = {meilleur_k}  (Accuracy = {accuracies[meilleur_k]:.4f})")

with open('meilleur_k.txt', 'w') as f:
    f.write(str(meilleur_k)) # sauvegarder le meilleur k afin de faciliter l'évaluation suivante

plt.figure(figsize=(6, 4))
plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o', color='steelblue', linewidth=2)
plt.axvline(meilleur_k, color='tomato', linestyle='--', label=f'Meilleur K={meilleur_k}')
plt.xlabel('Valeur de K')
plt.ylabel('Accuracy (validation)')
plt.title('Sélection du K optimal')
plt.legend()
plt.tight_layout()
plt.savefig('knn_selection_k.png', dpi=150)
plt.close()
 
print("Fichiers sauvegardés : meilleur_k.txt, knn_selection_k.png")
print("Veuillez lancer maintenant : python 3_evaluation.py")