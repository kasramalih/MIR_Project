import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data_loader import ReviewLoader
from basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)
        train_dataset = ReviewDataSet(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.set_valid_dataloader(X_val, y_val)
        best_f1 = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            # f1_train = 0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                # f1_train += f1_score(yb, preds)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            # f1_train /= self.num_epochs
            # print(f"f1 score on train data {f1_train}")
            train_loss /= len(train_loader)

            if self.val_loader:
                eval_loss, _, _, f1_macro = self._eval_epoch(self.val_loader, self.model)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, F1 Score (Macro): {f1_macro:.4f}")

                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    self.best_model = self.model.state_dict()
                    torch.save(self.model.state_dict(),'deep_model.pt')

        self.model.load_state_dict(self.best_model)
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))  # Dummy labels
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []

        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(self.device)
                preds = self.model(xb)
                predictions.extend(torch.argmax(preds, dim=1).cpu().numpy())

        return predictions

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        eval_loss = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb)
                loss = self.criterion(preds, yb)
                eval_loss += loss.item()
                predicted_labels.extend(torch.argmax(preds, dim=1).cpu().numpy())
                true_labels.extend(yb.cpu().numpy())

        eval_loss /= len(dataloader)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        return eval_loss, predicted_labels, true_labels, f1_macro

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self
    
    def set_valid_dataloader(self, X_val, y_val):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        val_dataset = ReviewDataSet(X_val, y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader(file_path='/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB Dataset.csv')
    review_loader.load_data()
    review_loader.get_embeddings()
    X_train, X_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.25)

    nn_classifier = DeepModelClassifier(in_features=100, num_classes=2, batch_size=32, num_epochs=50)
    # model.load_state_dict(torch.load('deep_model.pt'))

    nn_classifier.set_test_dataloader(X_test, y_test)
    print('DEEP model starting to train!')
    nn_classifier.fit(X_train, y_train)

    # Print classification report
    print('DEEP model results:')
    report = nn_classifier.prediction_report(X_test, y_test)
    print(report)


#TODO create a .json file that stores best model for each classifier so you dont need to train them each time!
# save the f1 score and the path to the model to load the model!
# current score for deep model is 0.75! it is not learning much coz the score in the first epoch is 0.72
# 