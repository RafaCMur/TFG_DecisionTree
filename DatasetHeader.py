import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DatasetHeader:

    def __init__(self, path):
        self._loadDataMulticlass_NSLKDD(path)
        self._preprocessData()
        #self._transformToBinaryData(4)
        print("============ Data loaded and preprocessed ============")


    def printData(self):
        print("\n==== X_train: ====\n\n", self.X_train)
        print("\n==== y_train: ====\n\n", self.y_train)
        print("\n==== X_test: ====\n\n", self.X_test)
        print("\n==== y_test: ====\n\n", self.y_test)
        
        print(" =========== Data info ===========")
        classes = pd.unique(self.y_test)
        print("Classes: ", classes)
        print("Number of classes: ", len(classes))
        # Numero de columnas
        print("Number of columns: ", len(self.header))
        # Nombre de las columnas

        print("Número de muestras totales en test: ", len(self.y_test))
        print("Número de muestras totales en train: ", len(self.y_train))
        print("Número de muestras totales: ", len(self.y_test) + len(self.y_train))

        # Obtener el número de muestras por clase en train
        unique_classes_train, counts_train = np.unique(self.y_train, return_counts=True)
        total_samples_train = np.sum(counts_train)
        percentages_train = (counts_train / total_samples_train) * 100
        data_train = {'Clase': unique_classes_train, 'Número de muestras': counts_train, 'Porcentaje': percentages_train}
        df_train = pd.DataFrame(data_train)

        # Obtener el número de muestras por clase en test
        unique_classes_test, counts_test = np.unique(self.y_test, return_counts=True)
        total_samples_test = np.sum(counts_test)
        percentages_test = (counts_test / total_samples_test) 
        data_test = {'Clase': unique_classes_test, 'Número de muestras': counts_test, 'Porcentaje': percentages_test}
        df_test = pd.DataFrame(data_test)

        # Obtener el numero de muestras totales en test y train
        total_samples = total_samples_test + total_samples_train
        total_percentages = (counts_test + counts_train) * 100 / total_samples 
        data_total = {'Clase': unique_classes_test, 'Número de muestras': counts_test + counts_train, 'Porcentaje': total_percentages}
        df_total = pd.DataFrame(data_total)

        # Mostrar los resultados en una tabla
        print("\nNúmero de muestras por clase en train:")
        print(df_train.to_string(index=False))

        print("Número de muestras por clase en test:")
        print(df_test.to_string(index=False))

        print("\nNúmero de muestras totales: ")
        print(df_total.to_string(index=False))




    def _loadDataMulticlass_UNSW_NB15(self, path):
        self.header = pd.read_csv(path + '/header.csv')
        self.header = list(pd.Series(self.header.iloc[:,0]))

        train = pd.DataFrame(np.load(path + '/raw_train.npy'), columns=self.header)
        test = pd.DataFrame(np.load(path + '/raw_test.npy'), columns=self.header)

        self.X_train = train.drop(['attack_cat', 'service', 'proto', 'state'], axis=1)
        self.y_train = train['attack_cat']
        self.X_test = test.drop(['attack_cat', 'service', 'proto', 'state'], axis=1)
        self.y_test = test['attack_cat']
    

    def _loadDataMulticlass_NSLKDD(self, path):
        self.header = pd.read_csv(path + '/header.csv')
        self.header = list(pd.Series(self.header.iloc[:,0]))

        train = pd.DataFrame(np.load(path + '/raw_train.npy'), columns=self.header)
        test = pd.DataFrame(np.load(path + '/raw_test.npy'), columns=self.header)

        self.X_train = train.drop(['label'], axis=1)
        self.y_train = train['label']
        self.X_test = test.drop(['label'], axis=1)
        self.y_test = test['label']


    # Takes the y column (attack_cat in unsw-nb15) and transforms it into a binary column.
    # If the value is 'Normal', it will be 0. Otherwise, it will be 1.
    def _transformToBinaryData(self, normal_class):
        self.y_train = np.where(self.y_train == normal_class, 0, 1)  # normal_class is the index of 'Normal' in the attack_cat column
        self.y_test = np.where(self.y_test == normal_class, 0, 1)    # If the value is [normal_class], it will be 0. Otherwise, it will be 1.


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

        #First: fit and transform train
        self.X_train = scaler.fit_transform(self.X_train)

        #Second: only transform test, not fit
        self.X_test = scaler.transform(self.X_test)

        # Both y_train and y_test are converted to numpy arrays because X_train and X_test are numpy arrays due to the transformation
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)