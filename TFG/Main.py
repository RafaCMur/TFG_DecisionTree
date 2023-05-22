import Dataset as DS
import DatasetHeader as DSH
import EatingTree as ET
import numpy as np
import matplotlib.pyplot as plt
import yaml
import testLibrary as tl


def plotGraphic():
    print("========== PLOTTING RESULTS  ==========")
    plt.plot(np.load('results/random_samples.npy'), np.load('results/random_results.npy'), label='random criteria')
    plt.plot(np.load('results/max_samples.npy'), np.load('results/max_results.npy'), label='max criteria')
    plt.plot(np.load('results/diff_samples.npy'), np.load('results/diff_results.npy'), label='diff criteria')
    print ("Size of random_samples: ", np.load('results/random_samples.npy').shape)
    print ("Size of random_results: ", np.load('results/random_results.npy').shape)
    print ("Size of max_samples: ", np.load('results/max_samples.npy').shape)
    print ("Size of max_results: ", np.load('results/max_results.npy').shape)
    plt.ylabel('F1 score')
    plt.xlabel('Number of samples')
    plt.legend()
    plt.show()


def execute_eat_and_save_results(et):
    results, samples = et.eat("random")
    np.save('results/random_results.npy', results)
    np.save('results/random_samples.npy', samples)
    
    results, samples = et.eat("max")
    np.save('results/max_results.npy', results)
    np.save('results/max_samples.npy', samples)

    results, samples = et.eat("diff")
    np.save('results/diff_results.npy', results)
    np.save('results/diff_samples.npy', samples)


def main():
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    #ds = DS.Dataset(config['DS']['path'])
    ds = DSH.DatasetHeader(config['DSH']['path'])

    """et = ET.EatingTree(config['ET']['initial_step'], 
                       config['ET']['step'], 
                       config['ET']['stop'], 
                       config['ET']['stop_samples'], 
                       ds.X_train, ds.y_train, ds.X_test, ds.y_test, random_state=config['ET']['random_state'])"""
    
    #execute_eat_and_save_results(et)

    #plotGraphic()


if __name__ == "__main__":
    main()

