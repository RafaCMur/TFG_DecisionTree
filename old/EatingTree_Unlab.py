import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import testLibrary as tl

"""
The idea is to have a tree that is going to take the training dataset and eat it "N" by "N" times
until it reaches the end of the dataset. The tree will then spit out the results of the training
using a plot.
"""
class EatingTree_Unlab():


    def __init__(self, step, initial_step, stop, stop_samples, X_train, y_train, X_test, y_test):
        self.step = step    # Number of samples to eat at a time
        self.initial_step = initial_step
        self.results = []
        self.samples = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.total_samples = len(X_train)
        self.stop = stop
        self.stop_samples = stop_samples

    
    def testTree(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        print(f1_score(self.y_test, y_pred, average='weighted'))
        print(classification_report(self.y_test, y_pred))
        print(self.X_train.shape, self.y_train.shape)


    def eat(self):

        np.random.seed(42)

        print("========== STARTING TO EAT  ==========")

        # Loop through the data and take increasing samples
        indices = np.arange(self.total_samples)
        np.random.shuffle(indices)

        # Split the data into labeled and unlabeled
        samples_taken = self.initial_step
        self.X_train_lab, self.X_train_unlab, self.y_train_lab, self.y_train_unlab = self._split_and_unlabel(first_indices=indices[:samples_taken])

        # Calculate the number of steps for the progress bar
        num_steps = (self.total_samples + self.step - self.initial_step - 1) // self.step
        
        with tqdm(total=num_steps) as pbar:
            while samples_taken < self.total_samples:
                
                # Extract the corresponding samples from the dataset
                X_sample = self.X_train_lab
                y_sample = self.y_train_lab

                # Train the tree on the sample
                tree = DecisionTreeClassifier(random_state=42)
                tree.fit(X_sample, y_sample)

                # Test the tree on the test set
                y_pred = tree.predict(self.X_test)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                self.results.append(f1)
                self.samples.append(samples_taken)

                # Order the unlabeled samples by their distance to the curve. The closest ones will be labeled
                step_unlab_idx = self._maxMethod(tree)

                # Update X_train_lab, X_train_unlab, y_train_lab, y_train_unlab
                self.X_train_lab = np.concatenate((self.X_train_lab, self.X_train_unlab[step_unlab_idx]))
                self.y_train_lab = np.concatenate((self.y_train_lab, self.y_train_unlab.iloc[step_unlab_idx]))
                self.X_train_unlab = np.delete(self.X_train_unlab, step_unlab_idx, axis=0)
                self.y_train_unlab = pd.Series(np.delete(self.y_train_unlab.values, step_unlab_idx, axis=0))

                # Update the number of samples taken
                samples_taken += self.step
                samples_taken = min(samples_taken, self.total_samples)

                if self.stop:
                    if samples_taken > self.stop_samples:
                        break

                # Update the progress bar
                pbar.update(1)
        
        print("========== DONE EATING  ==========")
        # Save the results to a npy file
        np.save("results/ETU_samples.npy", self.samples)
        np.save("results/ETU_results.npy", self.results)


    def _take(self, indices):
        X_train_lab = self.X_train[indices]
        y_train_lab = self.y_train[indices]
        X_train_unlab = np.delete(self.X_train, indices, axis=0)
        y_train_unlab = np.delete(self.y_train, indices, axis=0)
        return X_train_lab, X_train_unlab, y_train_lab, y_train_unlab

    def _split_and_unlabel(self, first_indices):
        
        # Create a boolean mask with the same length as your dataset
        mask = np.zeros(len(self.X_train), dtype=bool)
        mask[first_indices] = True

        # Use the mask to index the labeled and unlabeled parts of your dataset
        X_train_lab = self.X_train[mask]
        y_train_lab = self.y_train[mask]
        X_train_unlab = self.X_train[~mask]
        y_train_unlab = self.y_train[~mask]

        return X_train_lab, X_train_unlab, y_train_lab, y_train_unlab
    

    # Consists of taking the max of the probabilities for each element in the unlabeled dataset
    def _maxMethod(self, tree):
        # Predict probabilities on the unlabeled data
        pred_proba = tree.predict_proba(self.X_train_unlab)

        print("=Shape of pred_proba=", pred_proba.shape)
        print("==> Number of 1's: ", tl.count_ones(array=pred_proba), "of ", pred_proba.shape[0]*pred_proba.shape[1])
        print("==> Number of 0's: ", tl.count_zeros(array=pred_proba), "of ", pred_proba.shape[0]*pred_proba.shape[1])
        print("==> Numbers between 1 or 0 [Total - (1's + 0's)]: ", pred_proba.shape[0]*pred_proba.shape[1] - (tl.count_zeros(array=pred_proba) + tl.count_ones(array=pred_proba)))
        max_proba = np.max(pred_proba, axis=1)

        print("==> Number of 1's: ", tl.count_ones(array=max_proba), "of ", len(max_proba))
        print("==> Numbers != 1: ", len(max_proba) - tl.count_ones(array=max_proba))

        # Find the indices of the unlabeled data with the lowest probabilities
        unlab_idx = np.argsort(max_proba)[::1]
        print("Samples to be taken....", max_proba[unlab_idx[:10]])
        step_unlab_idx = unlab_idx[:self.step]
        return step_unlab_idx
    
    # Consists of taking the difference of the maximum and the minimum probabilities for each element in the unlabeled dataset
    def _diffMethod(self, tree):
        # Predict probabilities on the unlabeled data
        pred_proba = tree.predict_proba(self.X_train_unlab)
        max_proba = np.max(pred_proba, axis=1)
        min_proba = np.min(pred_proba, axis=1)
        diff_proba = max_proba - min_proba

        # Find the indices of the unlabeled data with the highest probabilities
        unlab_idx = np.argsort(diff_proba)[::1]
        print("Samples to be taken....", diff_proba[unlab_idx[:10]])
        step_unlab_idx = unlab_idx[:self.step]
        return step_unlab_idx
        
    


