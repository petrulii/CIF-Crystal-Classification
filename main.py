import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Replace 'your_file.csv' with the actual file name
file_path = 'crystals-spacegroups-distances.csv'

# The 'distances' column contains space-separated values as strings, we can use a lambda function to split it into a list:
df = pd.read_csv(file_path, converters={'distances': lambda x: list(map(float, x.split()))})

# Display the DataFrame, CSV file should have columns: id, space_group, distances
print(df)

# Extract the 'distances' column as input data
dataX = df['distances'].tolist()

# Extract the 'space_group' column as class labels
#dataY = df['sg'].tolist()
label_encoder = LabelEncoder()
dataY = label_encoder.fit_transform(df['sg'].tolist())
y = torch.tensor(dataY, dtype=torch.long)

# Convert data to PyTorch tensors
X = torch.tensor(dataX, dtype=torch.float32)
y = torch.tensor(dataY, dtype=torch.long)  # Use torch.long for integer labels

# Define a simple neural network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Random Classifier
class RandomClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, X):
        return np.random.randint(0, self.num_classes, size=len(X))

# Always Most Common Class Classifier
class MostCommonClassClassifier:
    def __init__(self, most_common_class):
        self.most_common_class = most_common_class

    def predict(self, X):
        return np.full(len(X), self.most_common_class)

# Function to plot the loss
def plot_loss(all_losses, model_names):
    plt.figure()
    for i, losses in enumerate(all_losses):
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_names[i]}')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    plt.close()

# Function to plot accuracy
def plot_accuracy(all_accuracies, model_names):
    plt.figure()
    print("len(all_accuracies):", len(all_accuracies))
    print("len(model_names):", len(model_names))
    plt.boxplot(all_accuracies, labels=model_names)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig('model_accuracy_plot.png')
    plt.close()

# evaluate model using k-fold cross-validation
def evaluate_model(X, y, n_folds=5):
    all_accuracies = []  # list to store accuracies
    all_losses = []  # list to store losses during training

    # Create an instance of the model
    n, num_classes = X.shape[1], len(torch.unique(y))
    print(num_classes, "classes counted")
    my_NN = SimpleClassifier(n, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_NN.parameters(), lr=1e-4)

    model_names = ['Neural Network', 'Random Classifier', 'Most Common Class Classifier']
    classifiers = [
        my_NN,
        RandomClassifier(num_classes),
        MostCommonClassClassifier(most_common_class=-1)
    ]

    # Loop through classifiers
    for classifier_name, classifier in zip(model_names, classifiers):
        accuracies = []  # list to store accuracies for each fold

        # Reset model for each fold
        if isinstance(classifier, SimpleClassifier):
            classifier = SimpleClassifier(n, num_classes)
            optimizer = optim.SGD(classifier.parameters(), lr=1e-3, momentum=0.9)

        # prepare cross-validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for i, (train_ix, test_ix) in enumerate(kfold.split(X)):
            # select rows for train and test
            trainX, trainY, testX, testY = X[train_ix], y[train_ix], X[test_ix], y[test_ix]

            # create DataLoader for training and testing sets
            train_dataset = TensorDataset(trainX, trainY)
            test_dataset = TensorDataset(testX, testY)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # training the model or using a baseline classifier
            epoch_losses = []  # list to store losses during each epoch
            if isinstance(classifier, SimpleClassifier):
                for epoch in range(100):
                    batch_losses = []
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        predictions = classifier(inputs)
                        loss = criterion(predictions, labels)
                        loss.backward()
                        optimizer.step()
                        batch_losses.append(loss.item())
                    epoch_losses.append(np.mean(batch_losses))
                classifier.eval()
            elif isinstance(classifier, MostCommonClassClassifier):
                classifier.most_common_class = np.argmax(np.bincount(trainY.numpy()))  # Identify the most common class
                epoch_losses = None
            else:
                epoch_losses = None

            # evaluating the model or baseline classifier
            all_preds = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    if isinstance(classifier, SimpleClassifier):
                        outputs = classifier(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.numpy())
                    else:
                        all_preds.extend(classifier.predict(inputs.numpy()))

            acc = accuracy_score(testY, all_preds)
            accuracies.append(acc * 100.0)
            print(f'{classifier_name} - Fold {i + 1} accuracy: {acc * 100.0:.3f}%')

        all_accuracies.append(accuracies)

        # Plot the loss
        if isinstance(classifier, SimpleClassifier):
            plot_loss([epoch_losses], [classifier_name])

    # Plot accuracy for all models
    plot_accuracy(all_accuracies, model_names)

    return all_accuracies

# Run the evaluation
accuracies = evaluate_model(X, y)

# Summarize model performance
for i, (classifier_name, acc) in enumerate(zip(['Neural Network', 'Random Classifier', 'Most Common Class Classifier'], accuracies)):
    print(f'{classifier_name} - Accuracy: mean={np.mean(acc):.3f} std={np.std(acc):.3f}, n={len(acc)}')
