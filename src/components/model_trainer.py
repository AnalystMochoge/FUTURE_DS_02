# ==============================
# 1. Imports
# ==============================
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# 2. Load and Clean Data
# ==============================
df = pd.read_csv('notebook/data/lemmatized_tickets.csv')
df = df[df['clean_description'].notna()].reset_index(drop=True)
documents = df['clean_description'].dropna().tolist()

# ==============================
# 3. Feature Extraction (TF-IDF)
# ==============================
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Sample features:", tfidf_vectorizer.get_feature_names_out()[:20])

# ==============================
# 4. Topic Modeling (NMF)
# ==============================
n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_features = nmf_model.fit_transform(tfidf_matrix)
df['NMF_Topic'] = nmf_features.argmax(axis=1)
print("NMF feature matrix shape:", nmf_features.shape)

# Print top words per topic
feature_names = tfidf_vectorizer.get_feature_names_out()
for index, topic in enumerate(nmf_model.components_):
    print(f"\nTOPIC #{index + 1}:")
    print([feature_names[i] for i in topic.argsort()[-10:][::-1]])

# ==============================
# 5. Assign Topic Labels
# ==============================
topic_labels = {
    0: 'Software & Firmware Updates',
    1: 'Performance & Stability Issues',
    2: 'Account Access & Login Problems',
    3: 'Unresolved Support & Follow-ups',
    4: 'Data Security & Safety Concerns'
}
df['Topic_Label'] = df['NMF_Topic'].map(topic_labels)

# Optional: Plot ticket distribution by topic
# df['Topic_Label'].value_counts().plot(kind='bar', color='skyblue')
# plt.title('Ticket Distribution by Topic')
# plt.ylabel('Number of Tickets')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# ==============================
# 6. Train/Test Split
# ==============================
X = tfidf_matrix
y = df['NMF_Topic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ==============================
# 7. Model Training & Evaluation
# ==============================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

# ==============================
# 8. Save Best Model (Random Forest)
# ==============================
best_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
joblib.dump(best_model, 'ticket_classifier_model.pkl')
joblib.dump(tfidf_vectorizer, 'ticket_vectorizer.pkl')

# ==============================
# 9. Sample Prediction
# ==============================
loaded_model = joblib.load('ticket_classifier_model.pkl')
sample_ticket = ["I'm unable to log into my account and it keeps showing invalid credentials."]
sample_vector = tfidf_vectorizer.transform(sample_ticket)
predicted_topic = loaded_model.predict(sample_vector)[0]
print("Predicted Topic:", topic_labels[predicted_topic])

# ==============================
# 10. Confusion Matrix
# ==============================
y_pred = loaded_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
labels = list(topic_labels.values())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xticks(rotation=45)
plt.grid(False)
plt.show()