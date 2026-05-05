import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Chargement des données

train_df = pd.read_csv("files/imdb_train.csv")
val_df = pd.read_csv("files/imdb_validation.csv")
test_df = pd.read_csv("files/imdb_test.csv")


# 2. Conversion des labels

label_map = {"positive": 1, "negative": 0}
reverse_label_map = {1: "positive",0: "negative"} # pour rendre les fichiers de sortie plus lisibles

train_df["sentiment"] = train_df["sentiment"].map(label_map)
val_df["sentiment"] = val_df["sentiment"].map(label_map)
test_df["sentiment"] = test_df["sentiment"].map(label_map)

y_train = train_df["sentiment"]
y_val = val_df["sentiment"]
y_test = test_df["sentiment"]


# 3. TF-IDF Vectorisation

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["review"])
X_val = vectorizer.transform(val_df["review"])
X_test = vectorizer.transform(test_df["review"])


# 4. Recherche du meilleur paramètre alpha
# en évaluant plusieurs valeurs sur le corpus de validation

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

best_alpha = alphas[0]
best_val_score = -1

print("===== VALIDATION TUNING =====")

for a in alphas:
    model = MultinomialNB(alpha=a, fit_prior=True)
    model.fit(X_train, y_train)

    val_pred_tmp = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred_tmp)

    print(f"alpha={a} -> validation accuracy={acc:.4f}")

    if acc > best_val_score:
        best_val_score = acc
        best_alpha = a

print("\nBEST alpha:", best_alpha)


# 5. Final model training

best_nb = MultinomialNB(alpha=best_alpha, fit_prior=True)
best_nb.fit(X_train, y_train)


# =========================================================
# Fonction : affichage confusion matrix texte
# =========================================================

def print_confusion_matrix(cm, labels):
    """
    (immiter le format comme weka)
    Format :
        a    b   <-- classified as
      xx   xx |    a = positive
      xx   xx |    b = negative
    """

    print("\nConfusion Matrix (text format)\n")

    print("      ", end="")
    for lab in labels:
        print(f"{lab:>8}", end="")
    print("   <-- classified as")

    for i, lab in enumerate(labels):
        print(f"{cm[i,0]:8}{cm[i,1]:8} |    {lab} = {labels[i]}")


# =========================================================
# 6. Validation evaluation + export
# =========================================================

val_pred = best_nb.predict(X_val)

print("\n===== FINAL VALIDATION RESULTS =====")

print("Accuracy:", accuracy_score(y_val, val_pred))

print(
    classification_report(
        y_val,
        val_pred,
        target_names=["negative", "positive"]
    )
)

# ----- confusion matrix texte -----

val_cm = confusion_matrix(y_val, val_pred)

print_confusion_matrix(
    val_cm,
    ["negative", "positive"]
)

# ----- confusion matrix image -----

disp = ConfusionMatrixDisplay(
    confusion_matrix=val_cm,
    display_labels=["negative", "positive"]
)

disp.plot(cmap="Blues",
          colorbar=False)

plt.title("Validation Confusion Matrix")

plt.savefig(
    "validation_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----- export csv -----

val_output = val_df.copy()

val_output["true_label"] = (
    y_val.map(reverse_label_map)
)

val_output["pred_label"] = (
    pd.Series(val_pred).map(reverse_label_map)
)

val_output["correct"] = (
    val_output["true_label"]
    == val_output["pred_label"]
)

val_output.to_csv(
    "NB_val_predictions.csv",
    index=False,
    encoding="utf-8"
)


# =========================================================
# 7. Test evaluation + export
# =========================================================

test_pred = best_nb.predict(X_test)

print("\n===== TEST RESULTS =====")

print("Accuracy:", accuracy_score(y_test, test_pred))

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["negative", "positive"]
    )
)

# ----- confusion matrix texte ----------

test_cm = confusion_matrix(y_test, test_pred)

print_confusion_matrix(
    test_cm,
    ["negative", "positive"]
)

# ----- confusion matrix image -------

disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=["negative", "positive"]
)

disp.plot(cmap="Greens", colorbar=False)

plt.title("Test Confusion Matrix")

plt.savefig(
    "test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----- export csv -------

test_output = test_df.copy()

test_output["true_label"] = (
    y_test.map(reverse_label_map)
)

test_output["pred_label"] = (
    pd.Series(test_pred).map(reverse_label_map)
)

test_output["correct"] = (
    test_output["true_label"]
    == test_output["pred_label"]
)

test_output.to_csv(
    "NB_test_predictions.csv",
    index=False,
    encoding="utf-8"
)

print("\nFichiers exportés :")
print("- val_predictions.csv")
print("- test_predictions.csv")
print("- validation_confusion_matrix.png")
print("- test_confusion_matrix.png")