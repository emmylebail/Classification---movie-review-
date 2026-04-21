import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. Chargement des données
# =========================
train_df = pd.read_csv("files/imdb_train.csv")
val_df = pd.read_csv("files/imdb_validation.csv")
test_df = pd.read_csv("files/imdb_test.csv")

# =========================
# 2. Conversion des labels
# =========================
train_df["sentiment"] = train_df["sentiment"].map({"positive": 1, "negative": 0})
val_df["sentiment"] = val_df["sentiment"].map({"positive": 1, "negative": 0})
test_df["sentiment"] = test_df["sentiment"].map({"positive": 1, "negative": 0})

y_train = train_df["sentiment"]
y_val = val_df["sentiment"]
y_test = test_df["sentiment"]

# =========================
# 3. Vectorisation TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["review"])
X_val = vectorizer.transform(val_df["review"])
X_test = vectorizer.transform(test_df["review"])

# =========================
# 4. Hyperparameter tuning (alpha) avec validation
# =========================
alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

best_alpha = None
best_val_score = 0

print("===== VALIDATION TUNING =====")

for a in alphas:
    model = MultinomialNB(alpha=a, fit_prior=True)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)

    print(f"alpha={a} -> validation accuracy={acc:.4f}")

    if acc > best_val_score:
        best_val_score = acc
        best_alpha = a

print("\nBEST alpha:", best_alpha)

# =========================
# 5. Entraînement final avec meilleur alpha
# =========================
best_nb = MultinomialNB(alpha=best_alpha, fit_prior=True)
best_nb.fit(X_train, y_train)

# =========================
# 6. Évaluation sur validation (final check)
# =========================
val_pred = best_nb.predict(X_val)

print("\n===== FINAL VALIDATION RESULTS =====")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=["negative", "positive"]))

# =========================
# 7. Évaluation finale sur test (résultat officiel)
# =========================
test_pred = best_nb.predict(X_test)

print("\n===== TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["negative", "positive"]))