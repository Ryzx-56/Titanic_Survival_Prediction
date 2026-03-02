from DataPreprocessing import preprocess

df = preprocess("titanic.csv")

# Decide the Feautures and the Goal (Y)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size= 0.2,
  random_state= 42,
  stratify= y
)

# Scale the numbers . 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Logisitc Regression 
from sklearn.linear_model import  LogisticRegression

model = LogisticRegression(
  max_iter= 1000,
  random_state= 42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

#Evaluate model
from sklearn.metrics import classification_report, confusion_matrix

print("Train Accuracy:", model.score(X_train_scaled, y_train))
print("Test Accuracy:", model.score(X_test_scaled, y_test))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

# add graphs Confusion matrix Heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test_scaled,
    y_test
)

plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix(Logistic_Regression).png")
plt.show()

# Roc Curve graph
from sklearn.metrics import RocCurveDisplay

y_prob = model.predict_proba(X_test_scaled)[:,1]

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve")
plt.savefig("ROC_Curve_(Logistic_Regression).png")
plt.show()

# Feature importance ( Logistics Coefficients)
import pandas as pd

coef = model.coef_[0]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coef
}).sort_values(by="Coefficient")

importance.set_index("Feature", inplace=True)

importance.plot(kind="barh")
plt.title("Feature Importance")
plt.savefig("Feature_Importance(Logistic_Regression).png")
plt.show()
