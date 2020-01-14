import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import pandas as pd

def plot_loss(experiment_dir):
    train_errs_p, train_errs_n = [], []
    val_errs_p = []
    val_errs_n = []
    f = open(os.path.join(experiment_dir,"err.txt"), "r")
    for line in f.readlines()[1:]:
        train_err_p, train_err_n, val_err_p, val_err_n = line.rstrip().split(':')
        train_errs_p.append(float(train_err_p))
        train_errs_n.append(float(train_err_n))
        val_errs_p.append(float(val_err_p))
        val_errs_n.append(float(val_err_n))

    train_val_err(train_errs_p, val_errs_p, backend='sns', path=os.path.join(experiment_dir,"err_p.png"))
    train_val_err(train_errs_n, val_errs_n, backend='sns', path=os.path.join(experiment_dir,"err_n.png"))

def train_val_err(train, val, step_size=5, backend='sns', path='err.png'):
    epochs = list(range(0,len(train)*step_size,step_size))
    plt.figure(figsize=(8,4))

    if backend == 'sns':
        df = pd.DataFrame()
        df['err'] = np.concatenate((train,val),axis=0)
        df['name'] = np.concatenate((['train']*len(train),['val']*len(val)),axis=0)
        #print(df['name'])
        df['epochs'] = np.concatenate((epochs,epochs),axis=0)
        lp = sns.lineplot(x="epochs", y="err", hue="name", data=df)
        #lp.set(xlim=(0,epochs[-1]))
    else:
        plt.plot(epochs, train, color='blue', label='Val')
        plt.plot(epochs, val, color='green', label='Train')
        plt.xlabel('Epochs')
        plt.ylabel('Err')
        plt.legend()
        #plt.xlim([0,epochs[-1]])

    #plt.ylim([0,0.1])
    #plt.yscale('log')
    plt.savefig(path)
    #plt.show()
    #plt.close()

if __name__ == '__main__':
    train = []
    val = []

    f = open("output/results/err_n.txt", "r")
    for line in f.readlines()[2:]:
        train_err, val_err = line.rstrip().split(':')
        train.append(float(train_err))
        val.append(float(val_err))

    train_val_err(train, val, backend='sns', path='err_n.png')

    '''

    '''
