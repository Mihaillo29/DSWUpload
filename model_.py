import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# I have created a new feature name debt_to_income_ratio, The feature represents the ratio of an individual's debt payments to their income
# which is often a crucial factor in determining creditworthiness, loan approval
# the Logistic model is used for prediction on the test_data.xlsx with accuracy of 71 

# Base class for all models
class BaseModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load(self, filepath):
        """Load data from an Excel file and compute derived features."""
        data = pd.read_excel(filepath)
        data['debt_to_income_ratio'] = data['installment'] / (data['annual_inc'] + 1e-6)  # Create the derived feature
        
        features = ['cibil_score', 'annual_inc', 'int_rate', 'loan_amnt', 'debt_to_income_ratio']
        X = data[features]
        y = data['loan_status']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess(self, X_train, X_test): # Normalizing the features as the Some feature like loan_amount and Income would have some Outlier 
                                            # # exampe someone takes a loan of 7000 and some takes a loan of 70000 this need to be scaled
        self.preprocessor = StandardScaler() 
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        return X_train, X_test

    def train(self, X_train, y_train): #train the Model
        if self.model is None:
            raise NotImplementedError("Model is not defined.")
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test): # Test the model and generate evaluation summary.
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def predict(self, X): 
        if self.model is None:
            raise NotImplementedError("Model is not defined.")
        X_preprocessed = self.preprocessor.transform(X)
        return self.model.predict(X_preprocessed)

# Logistic Regression model class
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()

# Decision Tree
class DecisionTree(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()()

# Training pipeline
def training_pipeline(filepath):
    """Prepare and execute the training pipeline for both models."""
    models = [LogisticRegressionModel(), DecisionTree()]
    
    for model in models:
        print(f"\nTraining {model.__class__.__name__}...")
        X_train, X_test, y_train, y_test = model.load(filepath) # Load and preprocess data
        X_train, X_test = model.preprocess(X_train, X_test)
        
        # Train and test model
        model.train(X_train, y_train)
        accuracy, report = model.test(X_test, y_test)
        
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        
        # Save the Logistic Regression model after training
        if isinstance(model, LogisticRegressionModel):
            joblib.dump(model, 'logistic_regression_model.pkl')
            print("Logistic Regression model saved as 'logistic_regression_model.pkl'.")


def predict_on_test_data(filepath): # Load the saved Logistic Regression model and test on the test data

    model = joblib.load('logistic_regression_model.pkl')
    
    test_data = pd.read_excel(filepath) # Load and preprocess the test data
    test_data['debt_to_income_ratio'] = test_data['installment'] / (test_data['annual_inc'] + 1e-6)
    
    features = ['cibil_score', 'annual_inc', 'int_rate', 'loan_amnt', 'debt_to_income_ratio']
    X_new = test_data[features]
    y_true = test_data['loan_status']

    X_new_preprocessed = model.preprocessor.transform(X_new)     # Preprocess the test data (same scaling as training data)
    
    # Make predictions
    predictions = model.model.predict(X_new_preprocessed)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, predictions)
    print(f"Accuracy on test data: {accuracy}")


if __name__ == "__main__":
    
    filepath = "train_data.xlsx"  # Training on the training dataset
    training_pipeline(filepath)
    test_filepath = "test_data.xlsx"
    predict_on_test_data(test_filepath)  # Predicting on new test data

"""Training LogisticRegressionModel...
Accuracy: 0.7474165603975199
Classification Report:
              precision    recall  f1-score   support

           0       0.56      0.14      0.22      5917
           1       0.76      0.96      0.85     16824

    accuracy                           0.75     22741
   macro avg       0.66      0.55      0.54     22741
weighted avg       0.71      0.75      0.69     22741

Logistic Regression model saved as 'logistic_regression_model.pkl'.

Training DecisionTreeModel...
Accuracy: 0.6490040015830438
Classification Report:
              precision    recall  f1-score   support

           0       0.34      0.36      0.35      5917
           1       0.77      0.75      0.76     16824

    accuracy                           0.65     22741
   macro avg       0.55      0.56      0.55     22741
weighted avg       0.66      0.65      0.65     22741

Accuracy on test data: 0.7141040804257835"""

