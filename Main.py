import Dataset as DS
import DatasetHeader as DSH
import Dataset_arff as DSA
import EatingTree as ET
import numpy as np
import matplotlib.pyplot as plt
import yaml


def plotGraphic(methods):
    print("========== PLOTTING RESULTS  ==========")
    plt.figure()
    for method in methods:
        plt.plot(np.load('results/' + method + '_samples.npy'), np.load('results/' + method + '_results.npy'), label=method + ' criteria')
    plt.ylabel('F1 score')
    plt.xlabel('Number of samples')
    plt.legend()
    plt.show()


def execute_eat_and_save_results(et, methods):
    times = []
    for method in methods:
        results, samples, time_elapsed = et.eat(method)
        np.save('results/' + method + '_results.npy', results)
        np.save('results/' + method + '_samples.npy', samples)
        times.append(time_elapsed)
    print("Times: ", times)




def main():
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    #dsa = DSA.Dataset_arff("C:\\Users\\34640\\Documents\\Python_VSCode\\TFG_DecisionTree\\nslkdd_complete")
    #ds = DS.Dataset(config['DS']['path'])
    ds = DSH.DatasetHeader(config['DSH']['path'])

    #ds.printData()
    

    et = ET.EatingTree(config['ET']['initial_step'],
                       config['ET']['step'],
                       config['ET']['stop'],
                       config['ET']['stop_samples'],
                       ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                       random_state=config['ET']['random_state'])
    
    execute_eat_and_save_results(et, config['methods'])

    plotGraphic(config['methods'])


if __name__ == "__main__":
    main()
