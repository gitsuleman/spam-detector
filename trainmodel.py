# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(lowercase=True, stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, pipeline.predict(X_test))

# Save model and accuracy
with open('spam_model.pkl', 'wb') as f:
    pickle.dump((pipeline, accuracy), f)

print(f"✅ Model trained. Accuracy: {accuracy:.2f}")
