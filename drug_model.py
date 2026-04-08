# STEP 1: Import libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# STEP 2: Load dataset
df = pd.read_csv(r"C:\Users\SuryasimhaMoturi\OneDrive - Health Catalyst\Practice\AI_data\assignments\drug_interaction_project\drug_side_effect_dataset_real_100k.csv")

# Select required columns
df = df[['age', 'gender', 'condition', 'drugA', 'drugB']]

# Rename columns
df.columns = ['age', 'gender', 'condition', 'druga', 'drugb']

# STEP 3: CLEAN TEXT
def clean_text(x):
    x = str(x).lower()
    x = re.sub(r'[^a-z0-9 ]', '', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

for col in ['condition', 'druga', 'drugb']:
    df[col] = df[col].apply(clean_text)

# STEP 4: CREATE SMART SIDE EFFECTS (Pattern-based)
df['side_effects'] = np.where(
    ((df['age'] > 50) & (df['druga'] != df['drugb'])) |
    ((df['condition'].str.contains('diabetes|bp|heart'))),
    'yes',
    'no'
)

# STEP 5: Encode Gender
df['gender'] = df['gender'].str.lower().map({'female': 0, 'male': 1})

# STEP 6: Encode Side Effects
df['side_effects'] = df['side_effects'].map({'yes': 1, 'no': 0})

# STEP 7: Label Encoding
le_condition = LabelEncoder()
le_druga = LabelEncoder()
le_drugb = LabelEncoder()

df['condition'] = le_condition.fit_transform(df['condition'])
df['druga'] = le_druga.fit_transform(df['druga'])
df['drugb'] = le_drugb.fit_transform(df['drugb'])

# STEP 8: Feature Engineering (IMPORTANT BOOST)
df['drug_interaction'] = df['druga'] * df['drugb']
df['age_risk'] = df['age'] // 10

# STEP 9: Define X and y
X = df[['age', 'gender', 'condition', 'druga', 'drugb', 'drug_interaction', 'age_risk']]
y = df['side_effects']

# STEP 10: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# STEP 11: Improved Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# STEP 12: Predictions
y_pred = model.predict(X_test)

# STEP 13: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("✅ Precision:", round(precision * 100, 2), "%")
print("✅ Recall:", round(recall * 100, 2), "%")
