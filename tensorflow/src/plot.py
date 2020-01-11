import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np

def plot_loss(histories, name, key='loss'):
    plt.figure(figsize=(8,4))

    for label, history in histories:
        plt.plot(history.epoch, history.history['val_'+key], color='blue', label=label.title()+' Val')
        plt.plot(history.epoch, history.history[key], color='green', label=label.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([0,0.1])
    #plt.yscale('log')
    plt.savefig('plots/pose_loss_{}.png'.format(name))
    #plt.show()
    plt.close()

def plot_history(histories, result_dir, exp_name, key='loss'):
    plt.figure(figsize=(6,4))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       color='blue', linestyle='--', label=name.title()+' Val')
        val_min = min(history.history['val_'+key])
        val_max = max(history.history['val_'+key])

        tra = plt.plot(history.epoch, history.history[key],
                       color=val[0].get_color(), label=name.title()+' Train')
        train_min = min(history.history[key])
        train_max = max(history.history[key])

    #plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.yscale('log')
    plt.xlim([0,max(history.epoch)])

    # find order of magnitude for min value on y axis
    min_ofm = math.floor(math.log10(min([val_min, train_min])))
    max_ofm = math.ceil(math.log10(max([val_max, train_max])))
    plt.ylim([10**min_ofm,10**max_ofm])

    plt.grid()
    plt.savefig(os.path.join(result_dir, exp_name+'_loss.png'))
    plt.close()

    #plt.show()

'''
def plot_history(history, result_dir):
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.plot(history.history['val_center_mean_absolute_error'], marker='o')
    plt.plot(history.history['val_norm_mean_absolute_error'], marker='o')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.yscale('log')

    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

def plot_history(history, result_dir):
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.yscale('log')

    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()
'''
