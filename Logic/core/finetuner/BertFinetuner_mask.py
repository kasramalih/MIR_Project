#imports
import json
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = None
        self.filtered_dataset = None
        self.top_genres = None
        self.label2id = None
        self.id2label = None
        self.trainer = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            self.dataset = json.load(file)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        #shall I remove other genres from the movies??
        self.load_dataset()
        print(len(self.dataset))
        genre_counter = Counter([genre for movie in self.dataset for genre in movie['genres']])
        self.top_genres = [genre for genre, _ in genre_counter.most_common(self.top_n_genres)]
        print(self.top_genres)
        filtered_dataset = [movie for movie in self.dataset if any(genre in self.top_genres for genre in movie['genres'])]
        self.filtered_dataset = filtered_dataset
        print(len(self.filtered_dataset))
        genre_counts = Counter([genre for movie in self.filtered_dataset for genre in movie['genres']])
        """
        plt.bar(genre_counts.keys(), genre_counts.values())
        plt.xlabel('Genre')
        plt.ylabel('Frequency')
        plt.title('Top {} Genre Distribution'.format(self.top_n_genres))
        plt.show()
        """
        self.label2id = {genre: idx for idx, genre in enumerate(self.top_genres)}
        self.id2label = {idx: genre for genre, idx in self.label2id.items()}
        print(self.label2id)
        print(self.id2label)

    def find_non_string_indices(self, input_list):
        if not isinstance(input_list, list):
            raise ValueError("Input must be a list")
        return [index for index, item in enumerate(input_list) if not isinstance(item, str)]

    def remove_indices_in_place(self, input_list, indices_to_remove):
        for index in sorted(indices_to_remove, reverse=True):
            del input_list[index]

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        summaries = [movie['first_page_summary'] for movie in self.filtered_dataset]
        genres = [movie['genres'] for movie in self.filtered_dataset]
        print(len(summaries), len(genres))
        non_str_indices = self.find_non_string_indices(summaries)   
        self.remove_indices_in_place(summaries, non_str_indices)
        self.remove_indices_in_place(genres, non_str_indices)
        print(len(summaries), len(genres))
        # Filter genres to top genres and convert to label ids
        # Convert genres to binary format
        labels = []
        for movie_genres in genres:
            label = [0] * len(self.top_genres)
            for genre in movie_genres:
                if genre in self.label2id:
                    label[self.label2id[genre]] = 1
            labels.append(label)
        print(len(summaries), len(labels))
        for i in range(10):
            print(summaries[i])
            print(labels[i])
            print('*'*20)

        token_lengths = [len(self.tokenizer(summary)['input_ids']) for summary in summaries]


        """
        plt.hist(token_lengths, bins=50)
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Lengths')
        plt.show()
        """

        print(f"Mean length: {np.mean(token_lengths)}")
        print(f"Median length: {np.median(token_lengths)}")
        print(f"Max length: {np.max(token_lengths)}")
        encodings = self.tokenizer(summaries, padding="max_length", truncation=True, max_length=64, return_tensors='pt')
        
        print(encodings.items())

        dataset = self.create_dataset(encodings, labels)
        test_size = int(test_size * len(dataset))
        val_size = int(val_size * (len(dataset) - test_size))
        train_size = len(dataset) - test_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        return train_dataset, val_dataset, test_dataset

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self,train_dataset, val_dataset, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=self.top_n_genres,
                                                           id2label=self.id2label,
                                                           label2id=self.label2id)
        
        metric_name = "f1"

        args = TrainingArguments(
            f"bert-finetuned-sem_eval-english",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps, 
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            #push_to_hub=True,
        )

        trainer = Trainer(
            model=model,                         
            args=args,                  
            train_dataset=train_dataset,         
            eval_dataset=val_dataset,            
            compute_metrics=self.compute_metrics,
        )

        self.trainer = trainer

        trainer.train()

    def multi_label_metrics(self, predictions, labels, threshold=0.3):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_macro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result

    def evaluate_model(self, test_dataset):
        """
        Evaluate the fine-tuned model on the test set.
        """
        """
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.top_n_genres)
        training_args = TrainingArguments(
            per_device_eval_batch_size=16,
            output_dir='./results'
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )
        """
        
        return self.trainer.evaluate()

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=self.top_n_genres,
                                                           id2label=self.id2label,
                                                           label2id=self.label2id)
        model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float) 

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """

        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)

bertf = BERTFinetuner('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB_crawled_given.json',5)
bertf.preprocess_genre_distribution()
train_dataset, val_dataset, test_dataset = bertf.split_dataset()

bertf.fine_tune_bert(train_dataset, val_dataset)

results = bertf.evaluate_model(test_dataset)
print(results)
bertf.save_model('my_finetuned_bert_model')

text = "Thrilling fight scenes and suspenseful music keep you entertained in this summer flick" 
#expected: action

encoding = bertf.tokenizer(text, return_tensors="pt")
encoding = {k: v.to(bertf.trainer.model.device) for k,v in encoding.items()}
outputs = bertf.trainer.model(**encoding)
logits = outputs.logits
logits.shape
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
predicted_labels = [bertf.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)