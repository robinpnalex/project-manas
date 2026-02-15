import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("gene_expression.csv")

print("Shape:", df.shape)
print(df["Cancer Present"].value_counts())

X = df.drop("Cancer Present", axis=1).values
y = df["Cancer Present"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

preds = model.predict(X_val)
print(f"\nAccuracy: {model.score(X_val, y_val):.4f}")
print(f"Support vectors per class: {model.n_support_}")
print(classification_report(y_val, preds, target_names=["No Cancer", "Cancer"]))
print("Confusion matrix:")
print(confusion_matrix(y_val, preds))
