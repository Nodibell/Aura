import os
import sys
# Disable Metal API validation layer for PyTorch MPS subprocess execution to prevent assertion crashes when run from Xcode
os.environ["MTL_DEBUG_LAYER"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# MPS has an unresolved Metal assertion bug (div_true_dense_scalar / read-only bytes
# bound with write access) when PyTorch runs inside Xcode's subprocess sandbox.
# Force CPU unconditionally so training is always stable.
_DEVICE = torch.device("cpu")

class TabularNN(nn.Module):
    """
    A lightweight, robust Deep Learning Neural Network for Tabular Data
    utilizing Residual Skip Connections, LayerNorm, and Dropout.
    """
    def __init__(self, input_dim, output_dim, is_classification):
        super().__init__()
        self.is_classification = is_classification
        
        # Initial transformation layer
        self.input_layer = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        
        # Residual Block 1
        self.hidden1 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        
        # Transition/Bottleneck Block 2
        self.hidden2 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        
        # Final output layer
        self.output_layer = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # Input layer
        x = self.dropout(self.relu(self.ln1(self.input_layer(x))))
        
        # First residual block (skip connection)
        residual = x
        x_res = self.dropout(self.relu(self.ln2(self.hidden1(x))))
        x = x + x_res # skip connection
        
        # Second hidden layer
        x = self.dropout(self.relu(self.ln3(self.hidden2(x))))
        
        # Output layer
        return self.output_layer(x)

class TabularNNClassifier:
    """
    Scikit-learn compatible wrapper for TabularNN Classification.
    """
    def __init__(self, epochs=30, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.classes_ = None
        self.device = _DEVICE
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Encode targets
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        
        # Convert to PyTorch tensors
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_encoded, dtype=torch.long)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        
        input_dim = X.shape[1]
        output_dim = len(self.classes_)
        
        self.model = TabularNN(input_dim, output_dim, is_classification=True).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            preds_encoded = torch.argmax(outputs, dim=1).cpu().numpy()
        return self.le.inverse_transform(preds_encoded)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

class TabularNNRegressor:
    """
    Scikit-learn compatible wrapper for TabularNN Regression.
    """
    def __init__(self, epochs=30, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.device = _DEVICE
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        
        input_dim = X.shape[1]
        output_dim = 1
        
        self.model = TabularNN(input_dim, output_dim, is_classification=False).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            preds = outputs.cpu().numpy().flatten()
        return preds

def train_and_evaluate_tabular_nn(X_train, y_train, X_test, y_test, is_classification, epochs=30, batch_size=64):
    """
    Trains a Tabular Neural Network and returns evaluation metric score,
    test set predictions, and the fitted wrapper model.
    Falls back to CPU if any device-level exception occurs.
    """
    # Prevent XGBoost/PyTorch OpenMP deadlock on macOS.
    # Both libraries ship their own libomp; when XGBoost runs n_jobs=-1 first
    # it holds the OpenMP init mutex. If PyTorch then tries to spawn its own
    # thread pool it deadlocks. Forcing single-threaded mode avoids the race.
    torch.set_num_threads(1)

    def _run(device_override=None):

        if is_classification:
            sys.stderr.write("Initializing TabularNN Classifier wrapper...\n")
            model = TabularNNClassifier(epochs=epochs, batch_size=batch_size)
            if device_override:
                model.device = device_override
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = float(accuracy_score(y_test, preds))
        else:
            sys.stderr.write("Initializing TabularNN Regressor wrapper...\n")
            model = TabularNNRegressor(epochs=epochs, batch_size=batch_size)
            if device_override:
                model.device = device_override
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = float(r2_score(y_test, preds))
        return score, preds, model

    try:
        return _run()
    except Exception as e:
        # If anything goes wrong (e.g., future device issues), retry on CPU
        cpu = torch.device("cpu")
        if _DEVICE != cpu:
            sys.stderr.write(f"TabularNN failed on {_DEVICE} ({e}), retrying on CPU...\n")
            return _run(device_override=cpu)
        raise
