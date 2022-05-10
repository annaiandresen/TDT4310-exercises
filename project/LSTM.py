import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.utils.data import (
    DataLoader, Dataset
)  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from sklearn.model_selection import train_test_split

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py
# https://www.youtube.com/watch?v=jGst43P-TJA

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256  # of rnn
num_classes = 10  # number of classes
learning_rate = 0.001
batch_size = 64
num_epochs = 2


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, hidden = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last hidden state to send to the linear layer

        return out

class TorchDataset(Dataset):
    def __init__(self, file_name, test_size):
        df = pd.read_csv(file_name, usecols=['text', 'label'])
        # Drop empty columns
        df = df.dropna()

        # Shuffle
        df.sample(frac=1).reset_index(drop=True)

        X = df['text'].values
        Y = df['label'].values
        self.test_size = test_size

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]



if __name__ == '__main__':
    ds = TorchDataset("data/dataset_small.csv", 0.2)
    train_loader = DataLoader(ds.x_train, batch_size=10, shuffle=False)
    test_loadeer = DataLoader(ds.y_test, batch_size=10, shuffle=False)
    print(next(train_loader.__iter__()))
    print(len(train_loader))
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    """
       # Initialize network
    model = BidirectionalLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()


    # Check accuracy on training & test to see how good our model

    def check_accuracy(loader, model):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy  \
                  {float(num_correct) / float(num_samples) * 100:.2f}"
            )

        model.train()


    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
    
    """

