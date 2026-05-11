# 🎬 Projet de Classification : Critiques de Films (Analyse de Sentiments)

Ce dépôt contient les travaux réalisés dans le cadre du cours de **Traitement Statistique des Données** (M1 PluriTAL, promotion 2025-2026).

Ce projet compare l'efficacité de quatre algorithmes de classification supervisée pour l'analyse de polarité (positif vs négatif) sur le corpus IMDB.

**Autrices :** LE BAIL Emmy, ZENG Youjia, ZHENG Ting  
**Référent :** Damien Nouvel 

---

## Objectif

L'enjeu est de déterminer lequel des algorithmes testés offre la meilleure précision pour distinguer une opinion **positive** d'une **négative**, en tenant compte des défis linguistiques (négations, richesse lexicale) propres aux critiques cinématographiques.

---

## Corpus

- **Source :** [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Maas et al.)
- **Volume :** 50 000 critiques équilibrées
- **Répartition :** 25 000 pour l'entraînement / 25 000 pour le test

---

## Méthodologie

### Prétraitement & Vectorisation

Le script `pretraitement.py` effectue les opérations suivantes :

- **Nettoyage :** Suppression des balises HTML, normalisation (minuscules) et retrait de la ponctuation.
- **Filtrage :** Suppression des *stopwords* (NLTK) en préservant les marqueurs de négation essentiels (*not*, *never*, *no*).
- **Représentation :**
  - TF-IDF via `TfidfVectorizer` (scikit-learn)
  - Filtre `StringToWordVector` pour les modèles WEKA

### Division des données (`split_dataset.py`)

Le dataset est scindé en trois parties pour garantir une évaluation rigoureuse :

| Ensemble | Part | Rôle |
| :--- | :---: | :--- |
| Entraînement | 70 % | Apprentissage |
| Validation | 15 % | Ajustement des hyperparamètres (K, alpha…) |
| Test | 15 % | Évaluation finale |

---

## Résultats et Comparaison

| Algorithme | Outil | Accuracy | F1-Score (pondéré) |
| :--- | :--- | :---: | :---: |
| **Naive Bayes** | scikit-learn | **86,45 %** | **0,865** |
| **Random Forest** | WEKA | 84,49 % | 0,845 |
| **KNN (K=345)** | scikit-learn | 83,05 % | 0,830 |
| **J48 (tuned)** | WEKA | 77,60 % | 0,776 |

---

## Utilisation et Reproductibilité

### 1. Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/votre-repo.git
cd votre-repo

# Installer les dépendances Python
pip install -r requirements.txt

# Télécharger les ressources NLTK nécessaires
python3 -c "import nltk; nltk.download('stopwords')"
```

### 2. Préparation des données

Placez le fichier `IMDB Dataset.csv` à la racine du dépôt disponible [ici](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data), puis lancez le découpage :

```bash
python3 split_dataset.py
```

Vous obtiendrez alors les fichiers `imbd_test.csv`, `imdb_train.csv`, `imdb_validation.csv`


### 3. Exécution des modèles

**Naive Bayes :**
```bash
python3 NaiveBayes.py
```

**KNN :**
```bash
python3 knn/1_entrainement.py
python3 knn/2_validation.py
python3 knn/3_test.py
```

**WEKA (J48 & Random Forest) :**  
Charger les fichiers `.arff` présents dans `data/`

---

## Structure du dépôt

```text
.
├── data/               # Fichiers CSV (train/val/test) et formats ARFF pour WEKA
├── evaluation/         # Matrices de confusion (PNG) et rapports de classification
├── knn/                # Scripts d'entraînement, validation et test du KNN
├── NaiveBayes.py       # Implémentation du modèle Naive Bayes
├── pretraitement.py    # Fonctions de nettoyage et normalisation
├── split_dataset.py    # Script de division Train / Val / Test
├── requirements.txt    # Dépendances Python
├── .gitignore          # Fichiers exclus (données lourdes, caches)
├── PluriTAL_TSD.pdf        # Rapport final
└── README.md           # Documentation du projet
```