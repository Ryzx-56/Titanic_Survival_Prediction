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

# create random forest model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# using grid search to get good parameters cause i traied random forest and got bad accuracy 
test_param = {
  'n_estimators': [100,200,300],
  'max_depth': [5,7,10,None],
  'min_samples_split': [2,5,10],
  'min_samples_leaf': [1,2,5],
  'max_features': ['sqrt', 'log2']

}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
  estimator=rf,
  param_grid=test_param,
  cv=5,
  scoring='roc_auc',
  n_jobs=-1
)

grid.fit(X_train,y_train)
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_rf = grid.best_estimator_


# predict on the test features
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# evaluate.
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test,y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# get the features with importance
import pandas as pd

feature_importance = pd.Series(best_rf.feature_importances_,
                               index= X_train.columns).sort_values(ascending=False)
print(feature_importance)

# make graphs to visulization 
# feature importance graph
import matplotlib.pyplot as plt
feature_importance.plot(kind='barh')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.savefig("Feature_Importance(Random_Forest).png")
plt.show()

# Roc Curve Graph
from sklearn.metrics import  roc_curve
y_probs = best_rf.predict_proba(X_test)[:,1]

fpr,tpr,_ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], linestyle= '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.savefig("ROC_Curve(Random_Forest).png")
plt.show()

# confusion matrix heatmap. its to see how many it predicted correctly and wrongly etc
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, best_rf.predict(X_test))

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xticks([0,1], ["Not Survived", "Survived"])
plt.yticks([0,1], ["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
  for j in range(2):
    plt.text(j,i,cm[i,j], ha="center", va="center")

plt.savefig("Confusion_Matrix_Heatmap(Random_Forest).png")
plt.show()
