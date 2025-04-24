# ==============================
# 1. Imports
# ==============================
import pandas as pd
import joblib
from sklearn.utils import resample
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


# ================================================
# 3. Balance Dataset + Feature Extraction (TF-IDF)
# ================================================

documents = df['clean_description'].dropna().tolist()

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
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
print(df['Topic_Label'].value_counts())

# Optional: Plot ticket distribution by topic
# df['Topic_Label'].value_counts().plot(kind='bar', color='skyblue')
# plt.title('Ticket Distribution by Topic')
# plt.ylabel('Number of Tickets')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# ==============================
# 6. Downsampling + Train/Test Split
# ==============================

df_majority = df[df['Topic_Label']=='Software & Firmware Updates']
df_minority = df[df['Topic_Label']!='Software & Firmware Updates']

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=500, #Match to minority class size
    random_state=42
)

#Combine majority + minority
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced= df_balanced.sample(frac=1, random_state=42). reset_index(drop=True) #shuffle

# TF-IDF on balanced dataset
balanced_docs = df_balanced['clean_description'].dropna().tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(balanced_docs)



X = tfidf_matrix
y = df_balanced['NMF_Topic']
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
sample_ticket = ["What is My Account? I'm unable to find the option to perform the desired action."]
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


from sklearn.manifold import TSNE
import seaborn as sns

# ==============================
# 11. Visualize Topic Overlap
# ==============================

# Run t-SNE on NMF topic features
tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_features = tsne_model.fit_transform(nmf_features)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame(tsne_features, columns=['x', 'y'])
tsne_df['Topic'] = df['Topic_Label']

# Plot using seaborn
plt.figure(figsize=(10, 7))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='Topic', palette='tab10', s=60, alpha=0.7)
plt.title('t-SNE Visualization of Topics (NMF)')
plt.legend(title='Topic Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot Top 10 Keywords for each topic
import matplotlib.pyplot as plt
import seaborn as sns

# Keywords for each topic (from your NMF results)
topics = {
    "Software & Firmware Updates": ['update', 'software', 'device', 'changes', 'software update', 'changes device', 'update changes', 'related', 'updated', 'related update'],
    "Performance & Stability Issues": ['acts', 'facing intermittent', 'times acts', 'unexpectedly', 'intermittent times', 'intermittent', 'acts unexpectedly', 'facing', 'times', 'contact'],
    "Unresolved Support & Follow-ups": ['customer support', 'contacted customer', 'support remains', 'remains', 'remains unresolved', 'unresolved', 'contacted', 'customer', 'support', 'account'],
    "Data Security & Safety Concerns": ['data', 'ensure', 'ensure data', 'like ensure', 'data safe', 'safe', 'concerned', 'im concerned', 'concerned security', 'security like'],
    "Account Access & Login Problems": ['error', 'screen', 'error message', 'message', 'popping screen', 'message popping', 'popping', 'account', 'access', 'access account']
}

# Dummy weights for plotting (since real weights were not provided)
weights = [1.0, 0.9, 0.85, 0.82, 0.8, 0.78, 0.76, 0.75, 0.74, 0.73]

# Plotting
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
axs = axs.flatten()

for i, (topic, keywords) in enumerate(topics.items()):
    sns.barplot(x=weights, y=keywords, ax=axs[i], palette="viridis")
    axs[i].set_title(topic)
    axs[i].set_xlabel("Relevance Score")
    axs[i].set_ylabel("Keywords")

# Hide unused subplot if the number of topics is odd
for j in range(len(topics), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.show()
