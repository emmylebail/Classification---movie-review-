import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Chargement des données
train_df = pd.read_csv("files/imdb_train.csv")
val_df = pd.read_csv("files/imdb_validation.csv")
test_df = pd.read_csv("files/imdb_test.csv")

# =========================
# 2. label 转换
# =========================
train_df["sentiment"] = train_df["sentiment"].map({"positive": 1, "negative": 0})
val_df["sentiment"] = val_df["sentiment"].map({"positive": 1, "negative": 0})
test_df["sentiment"] = test_df["sentiment"].map({"positive": 1, "negative": 0})

# =========================
# 3. TF-IDF 向量化
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["review"])
X_val = vectorizer.transform(val_df["review"])
X_test = vectorizer.transform(test_df["review"])

y_train = train_df["sentiment"]
y_val = val_df["sentiment"]
y_test = test_df["sentiment"]

# =========================
# 4. 训练 Naive Bayes
# =========================
nb = MultinomialNB()
nb.fit(X_train, y_train)

# =========================
# 5. validation 评估（选模型用）
# =========================
val_pred = nb.predict(X_val)

print("===== VALIDATION RESULTS =====")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=["negative", "positive"]))

# =========================
# 6. test 评估（最终结果）
# =========================
test_pred = nb.predict(X_test)

print("===== TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["negative", "positive"]))