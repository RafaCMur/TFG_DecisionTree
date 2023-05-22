import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

"""
The idea is to have a tree that is going to take the training dataset and eat it "N" by "N" times
until it reaches the end of the dataset. The tree will then spit out the results of the training
using a plot.
"""
class EatingTree():


    def __init__(self, initial_step, step , stop, stop_samples, X_train, y_train, X_test, y_test):
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

        # Generate a list of indices and shuffle it
        indices = np.arange(self.total_samples)
        np.random.shuffle(indices)

        # Loop through the data and take increasing samples
        samples_taken = self.initial_step
        sample_indices = indices[:samples_taken]

        # Calculate the number of steps for the progress bar
        num_steps = (self.total_samples + self.step - self.initial_step - 1) // self.step
        
        with tqdm(total=num_steps) as pbar:
            while samples_taken < self.total_samples:
                
                # Extract the corresponding samples from the dataset
                X_sample = self.X_train[sample_indices]
                y_sample = self.y_train[sample_indices]

                # Train the tree on the sample
                tree = DecisionTreeClassifier(random_state=42)
                tree.fit(X_sample, y_sample)

                # Test the tree on the test set
                y_pred = tree.predict(self.X_test)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                self.results.append(f1)
                self.samples.append(samples_taken)

                samples_taken += self.step
                samples_taken = min(samples_taken, self.total_samples)
                sample_indices = indices[:samples_taken]

                if self.stop:
                    if samples_taken > self.stop_samples:
                        break

                # Update the progress bar
                pbar.update(1)
        
        # Ending time in minutes
        end_time = time.time()
        
        print("========== DONE EATING  ==========")

        # Save the results to a npy file
        np.save("results/ET_samples.npy", self.samples)
        np.save("results/ET_results.npy", self.results)
