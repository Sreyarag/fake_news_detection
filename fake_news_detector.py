import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
    # Load dataset
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    # Add labels
    fake["label"] = "FAKE"
    true["label"] = "REAL"

    # Combine + reset index
    df = pd.concat([fake, true]).reset_index(drop=True)

    # Drop missing values properly
    df = df.dropna(subset=["title", "text", "label"])

    # Features and labels
    X = df["title"] + " " + df["text"]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Models
    lr = LogisticRegression(max_iter=1000, C=2)
    nb = MultinomialNB()

    # Train
    lr.fit(X_train_tfidf, y_train)
    nb.fit(X_train_tfidf, y_train)

    # Predict
    lr_pred = lr.predict(X_test_tfidf)
    nb_pred = nb.predict(X_test_tfidf)

    # Accuracy
    lr_acc = accuracy_score(y_test, lr_pred)
    nb_acc = accuracy_score(y_test, nb_pred)

    print(f"LR Accuracy: {lr_acc * 100:.2f}%")
    print(f"NB Accuracy: {nb_acc * 100:.2f}%")

    # Select best
    if lr_acc > nb_acc:
        best_model = lr
        best_pred = lr_pred
        best_acc = lr_acc
        print("✅ Using Logistic Regression")
    else:
        best_model = nb
        best_pred = nb_pred
        best_acc = nb_acc
        print("✅ Using Naive Bayes")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)

    print(f"\n🔥 Final Accuracy: {best_acc * 100:.2f}%")
    print("🔍 Confusion Matrix:")
    print(cm)

    return best_model, vectorizer