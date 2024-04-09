
import math
from typing import List

import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name
            wandb.init(project='MIR PHASE 1', entity='kasramlh')

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        true_positives = 0
        false_positives = 0

        for i in range(len(predicted)):
            for item in predicted[i]:
                if item in actual[i]:
                    true_positives += 1
                else:
                    false_positives += 1

        if true_positives + false_positives == 0:
            return 0.0  

        precision = true_positives / (true_positives + false_positives)
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        true_positives = 0
        false_negatives = 0

        for i in range(len(actual)):
            for item in actual[i]:
                if item in predicted[i]:
                    true_positives += 1
                else:
                    false_negatives += 1

        if true_positives + false_negatives == 0:
            return 0.0  

        recall = true_positives / (true_positives + false_negatives)
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0
        recall = self.calculate_recall(actual, predicted)
        precision = self.calculate_precision(actual, predicted)
        if recall + precision == 0:
            return 0 
        f1 = 2 * recall * precision / (recall + precision)
        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        total_precision = 0.0
        num_queries = len(actual)

        for i in range(num_queries):
            true_positives = 0
            precision_sum = 0.0
            relevant_items = len(actual[i])

            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    true_positives += 1
                    precision = true_positives / (j + 1)  # Precision at this position
                    precision_sum += precision

            if relevant_items == 0:
                continue 

            average_precision = precision_sum / relevant_items
            total_precision += average_precision

        if num_queries == 0:
            return 0.0 

        AP = total_precision / num_queries
        return AP
    
# no difference with AP because we are taking ap over multiple queries, we shall take it over one only!

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        total_AP = 0.0
        num_queries = len(actual)

        for i in range(num_queries):
            true_positives = 0
            precision_sum = 0.0
            relevant_items = len(actual[i])

            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    true_positives += 1
                    precision = true_positives / (j + 1)  # Precision at this position
                    precision_sum += precision

            if relevant_items == 0:
                continue

            average_precision = precision_sum / relevant_items
            total_AP += average_precision

        if num_queries == 0:
            return 0.0  

        MAP = total_AP / num_queries
        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        for i in range(len(actual)):
            ideal_order = actual[i]
            predicted_order = predicted[i]
            gain = 0
            for j in range(len(predicted_order)):
                if predicted_order[j] in ideal_order:
                    relevance = len(ideal_order) - ideal_order.index(predicted_order[j])
                    gain += (2**relevance - 1) / (math.log2(j + 2)) # j + 1 if j = 1 .. m
            DCG += gain

        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for i in range(len(actual)):
            ideal_order = actual[i]
            predicted_order = predicted[i]
            DCG = 0
            for j in range(len(predicted_order)):
                if predicted_order[j] in ideal_order:
                    relevance = len(ideal_order) - ideal_order.index(predicted_order[j])
                    DCG += (2**relevance - 1) / (math.log2(j + 2))

            sorted_ideal_order = sorted(ideal_order, key=lambda item: len(ideal_order) - ideal_order.index(item), reverse=True)
            #ideal dcg
            IDCG = 0
            for j in range(len(sorted_ideal_order)):
                IDCG += (2**(len(ideal_order) - j) - 1) / (math.log2(j + 2))

            if IDCG == 0:
                continue 
            NDCG += DCG / IDCG

        if len(actual) == 0:
            return 0.0

        return NDCG / len(actual)
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        for i in range(len(actual)):
            # Check if the actual item is present in the predicted list
            if actual[i][0] in predicted[i]:
                # Find the index of the first occurrence of the actual item in predicted list
                rank = predicted[i].index(actual[i][0]) + 1
                # Update RR with the reciprocal of the rank
                RR += 1 / rank

        # Calculate average RR
        if len(actual) > 0:
            RR /= len(actual)

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0
        num_queries = len(actual)
        total_rr = 0.0

        for i in range(num_queries):
            query_actual = actual[i]
            query_predicted = predicted[i]

            rr = 0.0
            for j, item in enumerate(query_predicted, start=1):
                if item in query_actual:
                    rr = 1 / j  # Reciprocal rank
                    break  # Stop when first relevant item is found

            total_rr += rr

        if num_queries > 0:
            MRR = total_rr / num_queries

        return MRR
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)


evaltest = Evaluation('test')
actual = [
    ['batman arkham night', 'batman dark knight', 'arkham asylum', 'batman begins', 'the batman'],
    ['haha', 'hehe', 'hoho', 'kiki', 'koko'],
    ['q', 'w', 'e', 'r', 't'],
    ['r', 'a', 'w', 'b', 'c'],
    ['p', 'o', 'i', 'u', 'y'],
    ['h', 'j', 'k', 'l', 'o']
]
predicted = [
    ['batman arkham night', 'batman dark knight', 'arkham asylum', 'batman begins', 'the batman'],
    ['hehe', 'hoho', 'haha', 'koko', 'kiki'],
    ['a', 'b', 'c', 'd', 'e'],
    ['a', 'b', 'c', 'w', 'e'],
    ['j', 'k', 'h', 'p', 'o'],
    ['j', 'l', 'd', 'w', 'e']
]
evaltest.calculate_evaluation(actual, predicted) # works fine, wandb works fine too.