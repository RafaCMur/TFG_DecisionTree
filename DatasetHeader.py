import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DatasetHeader:

    def __init__(self, path):
        self._loadDataMulticlass(path)
        self._preprocessData()
        self._transformToBinaryData()
        print("============ Data loaded and preprocessed ============")


    def printData(self):
        print("\n==== X_train: ====\n\n", self.X_train)
        print("\n==== y_train: ====\n\n", self.y_train)
        print("\n==== X_test: ====\n\n", self.X_test)
        print("\n==== y_test: ====\n\n", self.y_test)


    def _loadDataMulticlass(self, path):
        header = pd.read_csv(path + '/header.csv')
        header = list(pd.Series(header.iloc[:,0]))

        train = pd.DataFrame(np.load(path + '/raw_train.npy'), columns=header)
        test = pd.DataFrame(np.load(path + '/raw_test.npy'), columns=header)

        self.X_train = train.drop(['attack_cat', 'service', 'proto', 'state'], axis=1)
        self.y_train = train['attack_cat']
        self.X_test = test.drop(['attack_cat', 'service', 'proto', 'state'], axis=1)
        self.y_test = test['attack_cat']

    # Take the y column (attack_cat) and transform it into a binary column.
    # If the value is 'Normal', it will be 0. Otherwise, it will be 1.
    def _transformToBinaryData(self):
        self.y_train = np.where(self.y_train == 5, 0, 1)  # 5 is the index of 'Normal' in the attack_cat column
        self.y_test = np.where(self.y_test == 5, 0, 1)    # If the value is 5, it will be 0. Otherwise, it will be 1.


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