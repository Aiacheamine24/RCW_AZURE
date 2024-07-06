# External Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle
# Import Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer

# Load Data
dataset = pd.read_csv('./private/files/heart.csv')

# Define model_data dictionary to store transformed data
model_data = {}
client_model_data = {}
dataFrame = pd.DataFrame()

"""
Sauvgarde sous cette forme
# Example usage with the provided data
data = {
  "Standard Scaler": {
    "Decision Tree": {
      "accuracy": 1.0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "auc_roc": 1.0
    },
    "Logistic Regression": {
      "accuracy": 0.8634146341463415,
      "precision": 0.8264462809917356,
      "recall": 0.9345794392523364,
      "f1_score": 0.8771929824561403,
      "auc_roc": 0.8601468624833111
    },
    ...
  },
  ...
}
"""

# Transformers
transformers = [
    ('Standard Scaler', StandardScaler()),
    ('Min-Max Scaler', MinMaxScaler()),
    ('Normalizer', Normalizer()),
    ('Binarizer', Binarizer())
]

# Transform Data
for transformer_name, transformer in transformers:
    # Copying the original dataset to avoid overwriting
    transformed_dataset = dataset.copy()
    
    # Apply transformer to features
    transformed_features = transformer.fit_transform(transformed_dataset.drop(columns=['target']))
    
    # Extracting target variable
    target_variable = transformed_dataset['target'].values
    
    # Splitting the transformed data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(transformed_features, target_variable, test_size=0.2, random_state=0)
    
    # Storing the transformed data along with other details
    model_data[transformer_name] = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'transformer': transformer,
        'transformer_name': transformer_name
    }
    client_model_data[transformer_name] = {
        'transformer': transformer,
        'transformer_name': transformer_name
    }
    dataFrame = pd.DataFrame(model_data)



# Classifiers
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Logistic Regression', LogisticRegression()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('AdaBoost', AdaBoostClassifier())
]

# Train Models
# Train every model with every transformer and get for each model AUC-ROC, Accuracy, Precision, Recall, F1-Scorefrom sklearn.metrics import roc_auc_score

# Train Models
for classifier_name, classifier in classifiers:
    for transformer_name, data in model_data.items():
        # Train the model
        model = classifier.fit(data['x_train'], data['y_train'])
        
        # Make predictions
        y_pred = model.predict(data['x_test'])
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(data['y_test'], y_pred)
        
        # Get classification report
        class_report = classification_report(data['y_test'], y_pred, output_dict=True)
        
        # Get accuracy, precision, recall, f1-score
        accuracy = np.mean(y_pred == data['y_test'])
        precision = class_report['1']['precision']
        recall = class_report['1']['recall']
        f1_score = class_report['1']['f1-score']
        
        # Get AUC-ROC
        auc_roc = roc_auc_score(data['y_test'], y_pred)

        # Store the model along with other details
        model_data[transformer_name][classifier_name] = {
            'model': model,
            'classifier_name': classifier_name,
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc
        }
        client_model_data[transformer_name][classifier_name] = {
            'model': model,
            'classifier_name': classifier_name,
            'transformer_name': transformer_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc
        }

print(client_model_data)

