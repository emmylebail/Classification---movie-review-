import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# 0. Prétraitement (aligné avec l'approche KNN)
# =========================================================

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def pretraitement(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# =========================================================
# 1. Chargement des données
# =========================================================

train_df = pd.read_csv("files/imdb_train.csv")
val_df   = pd.read_csv("files/imdb_validation.csv")
test_df  = pd.read_csv("files/imdb_test.csv")

print("Prétraitement en cours... (peut prendre quelques minutes)")
train_df["review"] = train_df["review"].apply(pretraitement)
val_df["review"]   = val_df["review"].apply(pretraitement)
test_df["review"]  = test_df["review"].apply(pretraitement)
print("Prétraitement terminé.")

# =========================================================
# 2. Conversion des labels
# =========================================================

label_map         = {"positive": 1, "negative": 0}
reverse_label_map = {1: "positive", 0: "negative"}

train_df["sentiment"] = train_df["sentiment"].map(label_map)
val_df["sentiment"]   = val_df["sentiment"].map(label_map)
test_df["sentiment"]  = test_df["sentiment"].map(label_map)

y_train = train_df["sentiment"]
y_val   = val_df["sentiment"]
y_test  = test_df["sentiment"]

# =========================================================
# 3. Validation tuning sur les n-grams et différents alpha
# =========================================================

ALPHA = 1.0

ngram_options = [(1,1), (1,2), (1,3)]

best_ngram     = ngram_options[0]
best_val_score = -1

print("===== VALIDATION TUNING =====")

for ng in ngram_options:

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,   # stopwords déjà retirés en prétraitement
        ngram_range=ng
    )

    X_train = vectorizer.fit_transform(train_df["review"])
    X_val   = vectorizer.transform(val_df["review"])

    model = MultinomialNB(alpha=ALPHA, fit_prior=True)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"ngram_range={ng} -> validation accuracy={acc:.4f}")

    if acc > best_val_score:
        best_val_score = acc
        best_ngram     = ng

print("\nBEST ngram_range:", best_ngram)

# =========================================================
# 4. Vectorisation finale avec meilleur n-gram
# =========================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=None,   # stopwords déjà retirés en prétraitement
    ngram_range=best_ngram
)

X_train = vectorizer.fit_transform(train_df["review"])
X_val   = vectorizer.transform(val_df["review"])
X_test  = vectorizer.transform(test_df["review"])

# =========================================================
# 5. Entraînement du modèle final
# =========================================================

best_nb = MultinomialNB(alpha=ALPHA, fit_prior=True)
best_nb.fit(X_train, y_train)

# =========================================================
# Fonction : affichage confusion matrix texte (format Weka)
# =========================================================

def print_confusion_matrix(cm, labels):
    print("\nConfusion Matrix (text format)\n")
    print("      ", end="")
    for lab in labels:
        print(f"{lab:>10}", end="")
    print("   <-- classified as")
    for i, lab in enumerate(labels):
        row = ""
        for j in range(len(labels)):
            row += f"{cm[i,j]:10}"
        print(f"{row} |    {lab}")

# =========================================================
# 6. Évaluation validation + export
# =========================================================

val_pred = best_nb.predict(X_val)

print("\n===== FINAL VALIDATION RESULTS =====")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=["negative", "positive"]))

val_cm = confusion_matrix(y_val, val_pred)
print_confusion_matrix(val_cm, ["negative", "positive"])

disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=["negative", "positive"])
disp.plot(cmap="Blues", colorbar=False)
plt.title("Validation Confusion Matrix")
plt.savefig("validation_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

val_output = val_df.copy()
val_output["true_label"] = y_val.map(reverse_label_map)
val_output["pred_label"] = pd.Series(val_pred).map(reverse_label_map)
val_output["correct"]    = val_output["true_label"] == val_output["pred_label"]
val_output.to_csv("NB_val_predictions.csv", index=False, encoding="utf-8")

# =========================================================
# 7. Évaluation test + export
# =========================================================

test_pred = best_nb.predict(X_test)

print("\n===== TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["negative", "positive"]))

test_cm = confusion_matrix(y_test, test_pred)
print_confusion_matrix(test_cm, ["negative", "positive"])

disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=["negative", "positive"])
disp.plot(cmap="Greens", colorbar=False)
plt.title("Test Confusion Matrix")
plt.savefig("test_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

test_output = test_df.copy()
test_output["true_label"] = y_test.map(reverse_label_map)
test_output["pred_label"] = pd.Series(test_pred).map(reverse_label_map)
test_output["correct"]    = test_output["true_label"] == test_output["pred_label"]
test_output.to_csv("NB_test_predictions.csv", index=False, encoding="utf-8")

print("\nFichiers exportés :")
print("- NB_val_predictions.csv")
print("- NB_test_predictions.csv")
print("- validation_confusion_matrix.png")
print("- test_confusion_matrix.png")
