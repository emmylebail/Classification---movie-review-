# Projet de Classification : Critiques de Films

## Contexte
Ce dépôt contient les scripts et les données de notre projet de fin de semestre pour le cours de **Traitement statistique des données** du Master Traitement Automatique des Langues (**pluriTAL**), année 2025-2026.

**Auteurs :** ZHENG Ting, ZENG Youjia, LE BAIL Emmy

## 🎯 Objectif
L'objectif de ce projet est d'entraîner des classifieurs par apprentissage automatique et de comparer les performances de différents algorithmes de classification sur une tâche d'analyse de sentiments (polarité positive vs négative) à partir de critiques de films. 

## Corpus
- **Source des données :** [Précisez ici d'où viennent les critiques : AlloCiné, IMDb, un dataset existant...]
- **Taille :** [Au moins 100 documents par classe, ex: 500 critiques positives, 500 critiques négatives]
- **Prétraitement :** Les textes sont encodés en UTF-8. Les indices évidents (comme la note sur 5 étoiles) ont été retirés pour ne pas biaiser l'apprentissage[cite: 51].

## Méthodologie et Algorithmes
Nous avons extrait les caractéristiques des textes (vectorisation, TF-IDF, etc.) et testé les algorithmes suivants pour comparer leurs performances:
- [ ] Naive Bayes
- [ ] KNN (K-Nearest Neighbors)
- [ ] WEKA / Random forest
*(Cochez ceux que vous avez implémentés, n'oubliez pas qu'il en faut au moins deux de cette liste !)*

## Utilisation et Reproductibilité 
*(Expliquez ici comment lancer vos scripts avec des instructions suffisantes pour que le professeur puisse les tester [cite: 33])*

1. Cloner le dépôt : `git clone https://github.com/votre-nom/votre-repo.git`
2. Installer les dépendances : `pip install -r requirements.txt` *(si vous utilisez scikit-learn, spacy, etc. [cite: 87, 88])*
3. Lancer la vectorisation : `python3 vectorisation.py chemin/du/corpus chemin/du/fichier/de/sortie` 
4. Lancer l'entraînement et l'évaluation : `python main.py`