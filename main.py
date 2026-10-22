import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ========== 1. LOAD DATA ==========
print("Loading Titanic dataset...")
df = pd.read_csv('Titanic-Dataset.csv')
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")

# ========== 2. EDA CHARTS ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Chart 1: Survival count
sns.countplot(x='Survived', data=df, ax=axes[0,0], palette='viridis')
axes[0,0].set_title('Survived (0=No, 1=Yes)')
axes[0,0].set_xticklabels(['Died', 'Survived'])

# Chart 2: Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Survival by Gender')

# Chart 3: Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[0,2], palette='viridis')
axes[0,2].set_title('Survival by Ticket Class')

# Chart 4: Age distribution
sns.histplot(df['Age'].dropna(), bins=30, ax=axes[1,0], color='teal')
axes[1,0].set_title('Age Distribution')

# Chart 5: Fare distribution
sns.histplot(df['Fare'], bins=30, ax=axes[1,1], color='coral')
axes[1,1].set_title('Fare Distribution')

# Chart 6: Survival by Embarked
sns.countplot(x='Embarked', hue='Survived', data=df, ax=axes[1,2], palette='viridis')
axes[1,2].set_title('Survival by Embarked Port')

plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150)
print("\n📊 Saved eda_charts.png")

# ========== 3. DATA PREPROCESSING ==========
print("\n--- Preprocessing ---")

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing) and unnecessary columns
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

print("After preprocessing:")
print(df.head())
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ========== 4. SPLIT DATA ==========
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ========== 5. TRAIN MODELS ==========
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_acc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, 
          target_names=['Died', 'Survived'], digits=4))
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n✅ Best Model: {best_name} (Accuracy: {best_acc:.4f})")

# ========== 6. CONFUSION MATRIX ==========
pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title(f'Confusion Matrix — {best_name}\nAccuracy: {best_acc:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("📊 Saved confusion_matrix.png")

# ========== 7. FEATURE IMPORTANCE ==========
if best_name == "Random Forest":
    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(8, 5))
    importance.plot(kind='barh', color='teal')
    plt.title('Feature Importance — Which factors predict survival?')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("📊 Saved feature_importance.png")

# ========== 8. SAVE MODEL ==========
joblib.dump({
    'model': best_model,
    'model_name': best_name,
    'accuracy': best_acc,
    'features': list(X.columns)
}, 'titanic_model.pkl')
print(f"\n💾 Saved titanic_model.pkl")
print("\n✅ DONE! Now run: streamlit run app.py")