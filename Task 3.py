import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn import tree

df = pd.read_csv("bank.csv", sep=";")
print("✅ Data loaded successfully!")
print(df.head())
print("Shape of dataset:", df.shape)

print("\nMissing values in each column:\n", df.isnull().sum())
df = df.dropna()

le = LabelEncoder()
df['y'] = le.fit_transform(df['y'])
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\n✅ After encoding:\n", df.head())
print("\nData types after encoding:\n", df.dtypes)

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

y_pred = model.predict(X_test)
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:\n", importance.sort_values('Importance', ascending=False))

plt.figure(figsize=(15, 10))
tree.plot_tree(model, filled=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()

joblib.dump(model, "bank_decision_tree_model.pkl")
print("\n💾 Model saved as 'bank_decision_tree_model.pkl'")
print("\n✅ All steps completed successfully!")
