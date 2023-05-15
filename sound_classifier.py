import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


class playlistClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(playlistClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        x = nn.functional.softmax(x, dim=1)
        return x

def load_trained_model(model_path, input_size, num_classes):
    model = playlistClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def train_sound_classifier():

    # Load training JSON data
    data = []
    with open('soundStats.json', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    print(f"Loaded {len(df)} samples.") 

    # Preprocess the training data
    X = df.drop('playlist', axis=1)
    y = df['playlist']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.long)

    print(f"Training set size: {len(X_train)}") 

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = playlistClassifier(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the neural network
    epochs = 50
    batch_size = 32

    print("Starting training...") 
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")  
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size]
            labels = y_train[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Training completed.")  
    torch.save(model.state_dict(), "trained_model.pth")

    return model, input_size, num_classes, encoder, scaler  # Return input_size and num_classes

if __name__ == "__main__":
    model, input_size, num_classes, encoder, scaler = train_sound_classifier()
