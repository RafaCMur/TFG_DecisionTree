import time
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
from tqdm import tqdm


class EatingTree:

    def __init__(self, initial_step, step , stop, stop_samples, X_train, y_train, X_test, y_test, random_state=42):
        self.INITIAL_STEP = initial_step
        self.STEP = step    # Number of samples to eat at a time
        self.X_TRAIN = X_train
        self.Y_TRAIN = y_train
        self.X_TEST = X_test
        self.Y_TEST = y_test
        self.TOTAL_SAMPLES = len(X_train)
        self.STOP = stop
        self.STOP_SAMPLES = stop_samples
        self.RND_ST = random_state
        self._first_step()
    

    # The first step is to take INITIAL_STEP samples from the unlabeled set and put them in the
    # labeled set.
    def _first_step(self):
        # Shuffle data
        # np.random.seed(self.RND_ST)                                                                        TODO: ¿Es necesario?
        indices, self.X_TRAIN, self.Y_TRAIN = self._shuffle(self.X_TRAIN, self.Y_TRAIN)

        # Take initial samples
        X_lab = np.empty((0, self.X_TRAIN.shape[1]))  # Assuming X_TRAIN is a 2D array
        y_lab = np.empty((0,))  # Assuming y_TRAIN is a 1D array

        X_unlab = self.X_TRAIN.copy()
        y_unlab = self.Y_TRAIN.copy()
        
        indices = self._take_random_samples(self.INITIAL_STEP, X_unlab)
        self.initial_X_lab, self.initial_y_lab, self.initial_X_unlab, self.initial_y_unlab = self._update_sets(indices, X_lab, y_lab, X_unlab, y_unlab)


    def eat(self, CRITERIA):
        # Initialize variables
        self.results = []
        self.samples = []
        X_lab = self.initial_X_lab
        y_lab = self.initial_y_lab
        X_unlab = self.initial_X_unlab
        y_unlab = self.initial_y_unlab
        samples_taken = self.INITIAL_STEP
        np.random.seed(self.RND_ST)

        # Calculate the number of steps for the progress bar
        num_steps = (self.TOTAL_SAMPLES - self.INITIAL_STEP - 1) // self.STEP

        print("========== STARTING TO EAT  ==========")
        print("Criteria: ", CRITERIA, "\n")

        start_time = time.time()
        
        with tqdm(total=num_steps) as pbar:
            while (samples_taken <= self.TOTAL_SAMPLES):

                # Train the tree on the current labeled samples
                #tree = DecisionTreeClassifier()
                #tree = KNeighborsClassifier(n_neighbors=3)
                tree = MLPClassifier()
                #tree = SVC(kernel='linear', probability=True)
                tree.fit(X_lab, y_lab)

                # Predict on the test set and save results
                y_pred = tree.predict(self.X_TEST)
                f1 = f1_score(self.Y_TEST, y_pred, average='weighted') #TODO use macro or average='weighted'?
                self.results.append(f1)
                self.samples.append(samples_taken)

                print("-------> Results saved. F1 score: " + str(f1) + " <-------")
                print("Remaining: ", X_unlab.shape[0], " samples")
                print("Processed: ", X_lab.shape[0], " samples")

                # Loop control
                if samples_taken == self.TOTAL_SAMPLES:
                    break

                # Take samples
                if CRITERIA == "random":
                    indices = self._take_random_samples(self.STEP, X_unlab)
                elif CRITERIA == "max":
                    indices = self._take_max_samples(self.STEP, X_unlab, model=tree)
                elif CRITERIA == "margin":
                    indices = self._take_margin_samples(self.STEP, X_unlab, model=tree)
                elif CRITERIA == "entropy":
                    indices = self._take_entropy_samples(self.STEP, X_unlab, model=tree)
                elif CRITERIA == "gini":
                    indices = self._take_gini_samples(self.STEP, X_unlab, model=tree)
                else:
                    raise ValueError("criteria should be other of the ones defined in the EatingTree class, above this error message")
                
                # Update the sets
                X_lab, y_lab, X_unlab, y_unlab = self._update_sets(indices, X_lab, y_lab, X_unlab, y_unlab)
                
                # Update the number of samples taken
                samples_taken += self.STEP
                samples_taken = min(samples_taken, self.TOTAL_SAMPLES)

                # Stop if needed
                if self.STOP and samples_taken > self.STOP_SAMPLES:
                    break     # TODO Si pongo 600 iniciales y que vaya de 200 en 200, y STOP_SAMPLES = 1965, se para en 1800

                pbar.update(1)

        end_time = time.time()
        print("========== FINISHED EATING  ==========")

        return self.results, self.samples, end_time - start_time


    def _shuffle(self, X_train, y_train):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        # Reorder X_train and y_train using the shuffled indices
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        return indices, X_train_shuffled, y_train_shuffled
    
    
    # Take the samples randomly
    def _take_random_samples(self, n, X_unlab):
        # Take random indices
        n = min(n, X_unlab.shape[0])
        print("    n: ", n)
        indices = np.random.choice(X_unlab.shape[0], n, replace=False)
        return indices
    

    # Take the most uncertain samples using the max probability
    def _take_max_samples(self, n, X_unlab, model):
        # Take random indices
        n = min(n, X_unlab.shape[0])
        print("    n: ", n)
        pred_proba = model.predict_proba(X_unlab)
        max_proba = np.max(pred_proba, axis=1)
        indices = np.argsort(max_proba)     # Order ascendingly, to take the LEAST conficence samples
        indices = indices[:n]               # Take the first n
        print ("Indices size: ", indices.shape)
        print ("First 10 probabilities:", max_proba[indices])
        print ("There are ", np.count_nonzero(max_proba == 1), " samples with max proba 1")
        print ("There are ", np.count_nonzero(max_proba < 0.99999), " samples with max proba < 0.99999")
        print ("Max_proba size", max_proba.shape)
        return indices
    

    # Take the samples with the smallest margin between the two highest probabilities
    def _take_margin_samples(self, n, X_unlab, model):
        # Take random indices
        n = min(n, X_unlab.shape[0])
        print("    n: ", n)
        
        pred_proba = model.predict_proba(X_unlab)
        
        # Sort probabilities and select the highest two
        sort_proba = np.sort(pred_proba, axis=1)
        max1_proba, max2_proba = sort_proba[:, -1], sort_proba[:, -2]
        
        # Calculate margin
        margin_proba = max1_proba - max2_proba
        
        # Select the n samples where the difference between the two highest probabilities is smallest
        indices = np.argsort(margin_proba)[:n]

        print ("First 10 probabilities:", margin_proba[indices])
        
        return indices



    # Take the samples with the largest entropy
    def _take_entropy_samples(self, n, X_unlab, model):
        # Take random indices
        n = min(n, X_unlab.shape[0])
        print("    n: ", n)
        pred_proba = model.predict_proba(X_unlab)
        e = entropy(pred_proba, axis=1)
        indices = np.argsort(e)[-n:]
        return indices
    

    # Take the samples with the largest Gini impurity
    def _take_gini_samples(self, n, X_unlab, model):
        # Compute the predicted probabilities
        pred_proba = model.predict_proba(X_unlab)

        # Calculate the Gini impurity for each instance
        gini_impurity = 1 - np.sum(pred_proba**2, axis=1)

        # Select the n instances with the highest Gini impurity
        indices = np.argsort(gini_impurity)[-n:]

        return indices




    # Update the sets given the indices of the samples to take
    def _update_sets(self, indices, X_lab, y_lab, X_unlab, y_unlab):
        # Take the samples
        X_sample = X_unlab[indices]
        y_sample = y_unlab[indices]

        # Add the samples to the labeled set
        X_lab = np.concatenate((X_lab, X_sample), axis=0)
        y_lab = np.concatenate((y_lab, y_sample), axis=0)

        # Remove the samples from the unlabeled set
        X_unlab = np.delete(X_unlab, indices, axis=0)
        y_unlab = np.delete(y_unlab, indices, axis=0)

        return X_lab, y_lab, X_unlab, y_unlab

# Dudas:
# Weighted vs macro => Mejor weighted
# Macro es si tengo 10000 de clase 0 y 100 de clase 1, y predigo todo 0, el macro es 0.5, pero el weighted es 0.99
