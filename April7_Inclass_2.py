from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create dummy classification dataset
X, y = make_classification(n_samples=200, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define base estimator
est = LogisticRegression()  # alternatives: SVC(), DecisionTreeClassifier()

# n_estimators = number of base models
# max_samples = fraction or int number of samples drawn with replacement
bag_model = BaggingClassifier(base_estimator=est, n_estimators=10, max_samples=0.8, random_state=42)

# train the bagging model
bag_model.fit(X_train, y_train)

# predict on test set
predictions = bag_model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
