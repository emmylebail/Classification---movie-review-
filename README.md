# Projet de Classification : Critiques de Films

## Contexte
Ce dépôt contient les scripts et les données de notre projet de fin de semestre pour le cours de **Traitement statistique des données** du Master Traitement Automatique des Langues (**pluriTAL**), année 2025-2026.

**Auteurs :** [Prénom 1 Nom 1], [Prénom 2 Nom 2], [Prénom 3 Nom 3]

## 🎯 Objectif
L'objectif de ce projet est d'entraîner des classifieurs par apprentissage automatique et de comparer les performances de différents algorithmes de classification sur une tâche d'analyse de sentiments (polarité positive vs négative) à partir de critiques de films[cite: 15]. 

## Corpus
- **Source des données :** [Précisez ici d'où viennent les critiques : AlloCiné, IMDb, un dataset existant...]
- [cite_start]**Taille :** [Au moins 100 documents par classe[cite: 20], ex: 500 critiques positives, 500 critiques négatives]
- [cite_start]**Prétraitement :** Les textes sont encodés en UTF-8[cite: 42]. [cite_start]Les indices évidents (comme la note sur 5 étoiles) ont été retirés pour ne pas biaiser l'apprentissage[cite: 51].

## Méthodologie et Algorithmes
Nous avons extrait les caractéristiques des textes (vectorisation, TF-IDF, etc.) et testé les algorithmes suivants pour comparer leurs performances:
- [ ] Naive Bayes
- [ ] KNN (K-Nearest Neighbors)
- [ ] WEKA / Random forest
*(Cochez ceux que vous avez implémentés, n'oubliez pas qu'il en faut au moins deux de cette liste !)*

## Utilisation et Reproductibilité 
*(Expliquez ici comment lancer vos scripts avec des instructions suffisantes pour que le professeur puisse les tester [cite: 33])*

1. Cloner le dépôt : `git clone https://github.com/votre-nom/votre-repo.git`
2. [cite_start]Installer les dépendances : `pip install -r requirements.txt` *(si vous utilisez scikit-learn, spacy, etc. [cite: 87, 88])*
3. [cite_start]Lancer la vectorisation : `python3 vectorisation.py chemin/du/corpus chemin/du/fichier/de/sortie` [cite: 68]
4. Lancer l'entraînement et l'évaluation : `python main.py`