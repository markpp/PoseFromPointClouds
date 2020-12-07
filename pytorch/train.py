import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import numpy as np
import json
import random

from src.plot import plot_loss
from src.models import FullSplit
from src.optimizer import Lookahead, RAdam, Ralamb
from config import data_source, out_dim

parser = argparse.ArgumentParser()
parser.add_argument('--output-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=201)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

local_path = args.output_path or os.path.join(os.getcwd(), "models")
num_epochs = args.num_epochs
batch_size = args.batch_size


# random sampling
def sample_N_random(x,n_points=1024):
    candiate_ids = [i for i in range(x.shape[0])]
    sel = []
    for _ in range(n_points):
        # select idx
        idx = random.randint(0,len(candiate_ids)-1)
        sel.append(candiate_ids[idx])
        # remove that idx from point_idx_options
        del candiate_ids[idx]
    return np.array(x[sel])


def sample_xs(X,mode='ra'):
    sample_x = []
    for x in X:
        if mode == 'ra':
            sample_x.append(sample_N_random(x))
    return np.array(sample_x)

def train(model, opt, dev, crit, X, Y, scheduler, alpha=0.5):
    scheduler.step()
    model.train()
    total_loss = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        if len(indices) != batch_size:
            continue
        data, label = X[indices].to(dev), Y[indices].to(dev)
        p, _, n, _ = model(data)
        p_loss = crit(p, label[:,:3])
        n_loss = crit(n, label[:,3:])
        loss = alpha*p_loss + (1-alpha)*n_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1
        epoch_loss = total_loss / num_batches

    return epoch_loss

def evaluate(model, dev, crit, X, Y):
    model.eval()
    p_total_error = 0
    n_total_error = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data, label = X[indices].to(dev), Y[indices].to(dev)
        p, _, n, _ = model(data)
        p_err = crit(p, label[:,:3])
        n_err = crit(n, label[:,3:])

        num_batches += 1

        p_total_error += p_err
        p_epoch_error = p_total_error / num_batches

        n_total_error += n_err
        n_epoch_error = n_total_error / num_batches

    return p_epoch_error, n_epoch_error


def train_loop(dir, dev, train_X, train_Y, val_X, val_Y):
    model = FullSplit(type="PointNet", output_dims=out_dim)
    model = model.to(dev)

    lr = 0.005
    opt = Lookahead(base_optimizer=RAdam(model.parameters(), lr=lr),k=5,alpha=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(num_epochs*0.65),int(num_epochs*0.85)], gamma=0.05)

    plot_file = open("{}/err.txt".format(dir),'w')
    plot_file.write("p_train_err:n_train_err:p_val_err:n_val_err\n")

    p_best_val_err = 999.9
    n_best_val_err = 999.9

    train_crit = nn.MSELoss()
    eval_crit = nn.L1Loss()
    for epoch in range(num_epochs):
        #train_X_ = torch.from_numpy(sample_xs(train_X))
        train_loss = train(model, opt, dev, train_crit, train_X, train_Y, scheduler)
        print('Epoch #{}, training loss {:.5f}'.format(epoch,train_loss))
        if epoch % 5 == 0:
            with torch.no_grad():
                p_train_err, n_train_err = evaluate(model, dev, eval_crit, train_X, train_Y)
                p_val_err, n_val_err = evaluate(model, dev, eval_crit, val_X, val_Y)
                #val_err = p_val_err + n_val_err
                #train_err = p_train_err + n_train_err
                plot_file.write("{:.5f}:{:.5f}:{:.5f}:{:.5f}\n".format(p_train_err, n_train_err, p_val_err, n_val_err))

                #if val_err < best_val_err:
                if p_val_err < p_best_val_err and n_val_err < n_best_val_err:
                    p_best_val_err, n_best_val_err = p_val_err, n_val_err
                    #torch.save(model.state_dict(),'{}/model.pth'.format(dir))
                    torch.save(model,'{}/best_model.pkl'.format(dir))
                    print('Winner!')
                print('Epoch #{}. Train err: p: {:.5f}, n: {:.5f}'.format(epoch, p_train_err, n_train_err))
                print('Validation err: p: {:.5f}(best: {:.5f}), n: {:.5f}(best: {:.5f})'.format(p_val_err, p_best_val_err, n_val_err, n_best_val_err))

    torch.save(model,'{}/final_model.pkl'.format(dir))
    plot_file.close()


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data #
    '''
    train_X = np.load('{}/{}_X.npy'.format(data_source,"train"),allow_pickle=True)
    train_X = sample_xs(train_X).astype('float32')
    train_Y = np.load('{}/{}_Y.npy'.format(data_source,"train"),allow_pickle=True)[:,:3+out_dim].astype('float32')
    val_X = np.load('{}/{}_X.npy'.format(data_source,"test"),allow_pickle=True)
    val_X = sample_xs(val_X).astype('float32')
    val_Y = np.load('{}/{}_Y.npy'.format(data_source,"test"),allow_pickle=True)[:,:3+out_dim].astype('float32')
    '''
    val_X = np.load('{}/{}_x_1024_ra.npy'.format(data_source,"val"),allow_pickle=True).astype('float32')
    val_Y = np.load('{}/{}_y.npy'.format(data_source,"val"),allow_pickle=True)[:,:3+out_dim].astype('float32')
    train_X = np.load('{}/{}_x_1024_ra.npy'.format(data_source,"train"),allow_pickle=True).astype('float32')
    train_Y = np.load('{}/{}_y.npy'.format(data_source,"train"),allow_pickle=True)[:,:3+out_dim].astype('float32')
    print(val_Y[:1])
    print(train_Y[:1])
    #split = 200
    #train_x, train_y = train_X[split:,:,:3], torch.from_numpy(train_Y[split:])
    #val_x, val_y = torch.from_numpy(sample_xs(train_X[:split,:,:3])), torch.from_numpy(train_Y[:split])

    train_x, train_y = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(train_Y)
    val_x, val_y = torch.from_numpy(val_X[:,:,:3]), torch.from_numpy(val_Y)
    print("# training samples {}".format(int(len(train_x))))
    train_loop(local_path, dev, train_x, train_y, val_x, val_y)

    # plot #
    plot_loss(local_path)
