import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Faster than SVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

import pickle



# 1. Load Data (with error handling)
try:
    df = pd.read_csv(r"I:\AVA Intern\Data Science\Data Science\Python-DS-Projects\IMDB_Sentiment_Projects\IMDB_Large_Dataset.csv")
    df = df.sample(1000)  # Smaller dataset for testing
    print("Data loaded. Samples:", len(df))
except Exception as e:
    print("Error loading file:", e)
    exit()

# 2. Preprocess
print("Unique sentiment values before mapping:", df['sentiment'].unique())

# Show rows with unexpected sentiment values
unexpected = df[~df['sentiment'].isin(['positive', 'negative'])]
if not unexpected.empty:
    print("Rows with unexpected sentiment values:\n", unexpected['sentiment'].value_counts())

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Drop rows where sentiment is NaN (i.e., unmapped)
df = df.dropna(subset=['sentiment'])

# Print if any NaNs remain
print("Any NaNs left in sentiment?", df['sentiment'].isna().any())

df['sentiment'] = df['sentiment'].astype(int)  # Ensure integer type
print("Unique sentiment values after mapping:", df['sentiment'].unique())

# 3. Split Data
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF (smaller features)
tfidf = TfidfVectorizer(max_features=1000)  # Reduced features
print("Transforming text...")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Train Model (faster alternative)
print("Training model...")
model = LogisticRegression(max_iter=1000)  # Faster training
model.fit(X_train_tfidf, y_train)
print("Model trained!")  # Confirm training completes

# 6. Evaluate
try:
    y_pred = model.predict(X_test_tfidf)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
except Exception as e:
    print("Evaluation failed:", e)


# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
