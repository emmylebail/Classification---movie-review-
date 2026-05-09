#  Script 3 — Évaluation finale sur le test set

import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# chargement
X_train = sp.load_npz('X_train.npz')
X_test  = sp.load_npz('X_test.npz')
y_train = pd.read_csv('y_train.csv')['sentiment']
y_test  = pd.read_csv('y_test.csv')['sentiment']

with open('meilleur_k.txt') as f:
    K = int(f.read())

print(f"K utilisé : {K}")

# Entraînement & prédiction
knn = KNeighborsClassifier(n_neighbors=K, metric='cosine', n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Résultats
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Matrice de confusion 
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Matrice de confusion (K={K})')
plt.tight_layout()
plt.savefig('knn_matrice_confusion.png', dpi=150)
plt.close()
 
print("Sauvegardée : knn_matrice_confusion.png")