import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset:

    def __init__(self, path):
        self._loadDataMulticlass(path)
        self._preprocessData()
        print("============ Data loaded and preprocessed ============")


    def printData(self):
        print("\n==== X_train: ====\n\n", self.X_train)
        print("\n==== y_train: ====\n\n", self.y_train)
        print("\n==== X_test: ====\n\n", self.X_test)
        print("\n==== y_test: ====\n\n", self.y_test)
        classes = pd.unique(self.y_test)
        print("Classes: ", classes)
        print("Number of classes: ", len(classes))
        print("Number of samples in test: ", len(self.y_test))
        print("Number of samples in train: ", len(self.y_train))
        #Numero de muestras que pertenecen a cada clase en test
        print("Number of samples per class in test: ", self.y_test.value_counts())
        #Numero de muestras que pertenecen a cada clase en porcentaje en test
        print("Number of samples per class in percentage in test: ", self.y_test.value_counts(normalize=True))
        #Numero de muestras que pertenecen a cada clase en train
        print("Number of samples per class in train: ", self.y_train.value_counts())
        #Numero de muestras que pertenecen a cada clase en porcentaje en train
        print("Number of samples per class in percentage in train: ", self.y_train.value_counts(normalize=True))


    def _loadDataBinary(self, path):
        train = pd.read_csv(path + '\\UNSW_NB15_testing-set.csv')
        test = pd.read_csv(path + '\\UNSW_NB15_training-set.csv')

        self.columns = train.columns

        self.X_train = train.drop(['attack_cat', 'label', 'id', 'service', 'proto', 'state'], axis=1)
        self.y_train = train['label']
        self.X_test = test.drop(['attack_cat', 'label', 'id', 'service', 'proto', 'state'], axis=1)
        self.y_test = test['label']
        

    def _loadDataMulticlass(self, path):
        train = pd.read_csv(path + '\\UNSW_NB15_testing-set.csv')
        test = pd.read_csv(path + '\\UNSW_NB15_training-set.csv')

        self.columns = train.columns

        self.X_train = train.drop(['attack_cat', 'label', 'id', 'service', 'proto', 'state'], axis=1)
        self.y_train = train['attack_cat']
        self.X_test = test.drop(['attack_cat', 'label', 'id', 'service', 'proto', 'state'], axis=1)
        self.y_test = test['attack_cat']

    """
    The data is reescaled or normalized because there are some columns with great values around 1 million and some of them with
    values around 10. It is more efficient if we readjust this data normalizing it to be in the same range more or less
    """
    def _preprocessData(self):
        # == Ordinal Encoding == From categorical to numerical
        #encoder = OrdinalEncoder()
        #self.X_train = encoder.fit_transform(self.X_train)
        #self.X_test = encoder.transform(self.X_test)

        # == Standardization ==
        scaler = StandardScaler()

        #First: fit and transform train, so that it goes -1...1
        self.X_train = scaler.fit_transform(self.X_train)

        #Second: only transform test, so that if there's a new non-classified value
        #out of the bounds -1...1, it gets standarized for example to 1.3432
        self.X_test = scaler.transform(self.X_test)

        # Both y_train and y_test are converted to numpy arrays because X_train and X_test are numpy arrays due to the transformation
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
