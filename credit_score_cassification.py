import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = {
    'Age': [25, 45, 35, 33, 50, 23, 40, 38, 29, 60],
    'Income': [50000, 120000, 80000, 60000, 150000, 45000, 100000, 95000, 52000, 200000],
    'Loan Amount': [10000, 20000, 15000, 12000, 30000, 8000, 18000, 17000, 11000, 40000],
    'Credit Score': [1, 2, 1, 1, 2, 0, 2, 2, 1, 2]  # 0: Bad, 1: Fair, 2: Good
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'Loan Amount']]
y = df['Credit Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))