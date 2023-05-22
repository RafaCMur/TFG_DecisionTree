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
class EatingTree_Unlab_mod():

    def __init__(self, initial_step, step , stop, stop_samples, X_train, y_train, X_test, y_test):
        self.STEP = step    # Number of samples to eat at a time
        self.INITIAL_STEP = initial_step
        self.results = []
        self.samples = []
        self.X_TRAIN = X_train
        self.Y_TRAIN = y_train
        self.X_TEST = X_test
        self.Y_TEST = y_test
        self.TOTAL_SAMPLES = len(X_train)
        self.stop = stop
        self.stop_samples = stop_samples

    
    def testTree(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X_TRAIN, self.Y_TRAIN)
        y_pred = tree.predict(self.X_TRAIN)
        print(f1_score(self.Y_TEST, y_pred, average='weighted'))
        print(classification_report(self.Y_TEST, y_pred))
        print(self.X_TRAIN.shape, self.Y_TRAIN.shape)


    def eat(self):

        np.random.seed(42)

        print("========== STARTING TO EAT  ==========")

        # Loop through the data and take increasing samples
        indices = np.arange(self.TOTAL_SAMPLES)
        np.random.shuffle(indices)

        # Split the data into labeled and unlabeled
        samples_taken = self.INITIAL_STEP
        lab_idx, unlab_idx = self._split(indices)
        if (tl.has_common_elements(lab_idx, unlab_idx)):
            print("ERROR: lab_idx and unlab_idx have COMMON elements")

        u_idx = np.arange(self.X_TRAIN[unlab_idx].shape[0])
        if(tl.has_uncommon_elements(unlab_idx, u_idx)):
            print("ERROR: self.X_train_unlab and unlab_idx have UNCOMMON elements: ", tl.count_uncommon_elements(u_idx, unlab_idx))

        self.print_sizes(lab_idx, unlab_idx, indices)


        # Calculate the number of steps for the progress bar
        num_steps = (self.TOTAL_SAMPLES + self.STEP - self.INITIAL_STEP - 1) // self.STEP
        
        with tqdm(total=num_steps) as pbar:
            while samples_taken < self.TOTAL_SAMPLES:
                
                # Extract the corresponding samples from the dataset
                X_sample = self.X_TRAIN[lab_idx]
                y_sample = self.Y_TRAIN[lab_idx]

                # Train the tree on the sample
                tree = DecisionTreeClassifier(random_state=42)
                print ("Number of classes: ", np.unique(y_sample, return_counts=True))
                tree.fit(X_sample, y_sample)

                # Test the tree on the test set
                y_pred = tree.predict(self.X_TEST)
                f1 = f1_score(self.Y_TEST, y_pred, average='weighted')
                self.results.append(f1)
                self.samples.append(samples_taken)

                # Order the unlabeled samples by their distance to the curve. The closest ones will be labeled
                step_unlab_idx = self._diffMethod(tree, lab_idx, unlab_idx)

                # Update the indexes
                lab_idx, unlab_idx = self._updateIndexes(lab_idx, unlab_idx, step_unlab_idx)

                # Update the number of samples taken
                samples_taken += self.STEP
                samples_taken = min(samples_taken, self.TOTAL_SAMPLES)

                if self.stop:
                    if samples_taken > self.stop_samples:
                        break

                # Update the progress bar
                pbar.update(1)
        
        print("========== DONE EATING  ==========")
        # Save the results to a npy file
        np.save("results/ETU_samples.npy", self.samples)
        np.save("results/ETU_results.npy", self.results)


    def _split(self, indices):
        lab_idx = indices[:self.INITIAL_STEP]
        unlab_idx = indices[self.INITIAL_STEP:]
        return lab_idx, unlab_idx
    
    def _updateIndexes(self, lab_idx, unlab_idx, step_unlab_idx):
        lab_idx = np.concatenate((lab_idx, step_unlab_idx))
        unlab_idx = np.setdiff1d(unlab_idx, step_unlab_idx)
        return lab_idx, unlab_idx



    # Consists of taking the max of the probabilities for each element in the unlabeled dataset
    def _maxMethod(self, tree, lab_idx, unlab_idx):
        if(tl.has_common_elements(lab_idx, unlab_idx)):
            print("ERROR: lab_idx and unlab_idx have COMMON elements")
        # Get the probabilities with the unlabeled set
        probs = tree.predict_proba(self.X_TRAIN[unlab_idx])
        # Get the max probability for each element
        max_probs = np.max(probs, axis=1)

        max_probs_df = pd.DataFrame({'original_index': unlab_idx, 'max_prob': max_probs})
        # Order the array by the max probability in ascending order
        sorted_max_probs_df = max_probs_df.sort_values(by='max_prob')  # Order by the max_prob values, not by the indexes
        # Get the indexes of the sorted array in ascending order
        sorted_idx = sorted_max_probs_df['original_index'].values

        if(tl.has_uncommon_elements(sorted_idx, unlab_idx)):
            print("ERROR: sorted_idx and unlab_idx have UNCOMMON elements: ", tl.count_uncommon_elements(sorted_idx, unlab_idx))
        
        print("==> First 10 elements of max_probs: \n", sorted_max_probs_df[:10])
        print("==> Numbers != 1: ", len(sorted_max_probs_df) - tl.count_ones(array=sorted_max_probs_df) )

        # Get the first self.step elements of the sorted array
        step_unlab_idx = sorted_idx[:self.STEP]
        if not tl.is_subset(subset=step_unlab_idx, superset=unlab_idx):
            print("ERROR: step_unlab_idx is not a subset of unlab_idx")
        
        return step_unlab_idx
    

    # Consists of taking the difference of the maximum and minimum probabilities for each element in the unlabeled dataset
    def _diffMethod(self, tree, lab_idx, unlab_idx):
        if(tl.has_common_elements(lab_idx, unlab_idx)):
            print("ERROR: lab_idx and unlab_idx have COMMON elements")
        # Get the probabilities with the unlabeled set
        probs = tree.predict_proba(self.X_TRAIN[unlab_idx])
        # Get the max probability for each element
        max_probs = np.max(probs, axis=1)
        min_probs = np.min(probs, axis=1)
        diff_probs = max_probs - min_probs

        diff_probs_df = pd.DataFrame({'original_index': unlab_idx, 'diff_prob': diff_probs})
        # Order the array by the difference probability in ascending order
        sorted_diff_probs_df = diff_probs_df.sort_values(by='diff_prob')  # Order by the diff_prob values, not by the indexes
        # Get the indexes of the sorted array in ascending order
        sorted_idx = sorted_diff_probs_df['original_index'].values

        if(tl.has_uncommon_elements(sorted_idx, unlab_idx)):
            print("ERROR: sorted_idx and unlab_idx have UNCOMMON elements: ", tl.count_uncommon_elements(sorted_idx, unlab_idx))
        
        print("==> First 10 elements of diff_probs: \n", sorted_diff_probs_df[:10])
        print("==> Numbers != 1: ", len(sorted_diff_probs_df) - tl.count_ones(array=sorted_diff_probs_df) )

        # Get the first self.step elements of the sorted array
        step_unlab_idx = sorted_idx[:self.STEP]
        if not tl.is_subset(subset=step_unlab_idx, superset=unlab_idx):
            print("ERROR: step_unlab_idx is not a subset of unlab_idx")
        
        return step_unlab_idx

        
    def print_sizes(self, lab_idx, unlab_idx, indices):
        print ("---------- Printing sizes ----------")
        print ("lab_idx size: ", len(lab_idx))
        print ("unlab_idx size: ", len(unlab_idx))
        print ("Total index size: ", len(lab_idx) + len(unlab_idx), "==", len(indices))
        print ("X_train_lab size: ", self.X_TRAIN[lab_idx].shape)
        print ("X_train_unlab size: ", self.X_TRAIN[unlab_idx].shape)
        print ("y_train_lab size: ", self.Y_TRAIN[lab_idx].shape)
        print ("y_train_unlab size: ", self.Y_TRAIN[unlab_idx].shape)
        print ("------------------------------------\n")
