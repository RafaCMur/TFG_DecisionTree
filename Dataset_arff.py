import pandas as pd


class Dataset_arff:

    def __init__(self, path):
        # Define the column names for the dataset
        column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"]

        # Load the training data
        train_data = pd.read_csv(path + "/KDDTrain+.txt", names=column_names)

        # Load the test data
        test_data = pd.read_csv(path + "/KDDTest+.txt", names=column_names)

        # Display the first few rows of the training data
        print(train_data.head())


        # Get the number of rows and columns in the training data
        train_rows, train_cols = train_data.shape
        print(f'Training data has {train_rows} rows and {train_cols} columns.')

        # Get the number of rows and columns in the test data
        test_rows, test_cols = test_data.shape
        print(f'Test data has {test_rows} rows and {test_cols} columns.')