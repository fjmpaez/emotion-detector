import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pickle
import argparse

def build_model(model_type: str):

    print(f"Building model of type: {model_type}")

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    return pipeline

def train_model(source_path, model_path, model_type):
    # Load the dataset
    df = pd.read_csv(source_path)
    
    df = df.dropna()

    features = df.drop('Class',axis=1)
    labels = df['Class']

    # Encode the labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Create a pipeline
    model_pipeline = build_model(model_type=model_type)


    # Fit the pipeline on the training data
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    yhat = model_pipeline.predict(X_test)
    print(yhat)
    model_performance = classification_report(y_test,yhat)
    print(f"Model Report: {model_performance}")

   # Save model and encoder
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model_pipeline,
            'label_encoder': label_encoder
        }, f)


if __name__ == "__main__":
    # Load the dataset
    parser = argparse.ArgumentParser(description="Train a model on the dataset.")
    parser.add_argument('--data', required=True, type=str, help='Path to the dataset.')
    parser.add_argument('--output', type=str, required=True, help='Path to the exported model.')
    parser.add_argument('--model_type', type=str, default="logistic_regression", choices=["logistic_regression", "random_forest"], help='Type of model to train.')
    
    args = parser.parse_args()
    source_path = args.data
    model_path = args.output    
    model_type = args.model_type

    train_model(source_path=source_path, model_path=model_path, model_type=model_type)
    print(f"Model trained and saved to {model_path}")
    print("Training completed.")    

    
    
