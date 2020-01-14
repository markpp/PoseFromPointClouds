import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import numpy as np
import json

from src.plot import plot_loss
from src.models import FullSplit
from src.optimizer import Lookahead, RAdam, Ralamb

parser = argparse.ArgumentParser()
parser.add_argument('--output-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=201)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

local_path = args.output_path or os.path.join(os.getcwd(), "models")
num_epochs = args.num_epochs
batch_size = args.batch_size

network_type = ["PointNet", "PointNet++", "GNN"][0]

def train(model, opt, dev, crit, X, Y, scheduler, alpha=0.95):
    scheduler.step()
    model.train()
    total_loss = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
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
    model = FullSplit(type="PointNet")
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
    data_source = "linemod"

    val_X = np.load('{}/{}/{}_x_1024_ra.npy'.format("input",data_source,"val"),allow_pickle=True).astype('float32')
    val_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"val"),allow_pickle=True)[:,:].astype('float32')

    train_X = np.load('{}/{}/{}_x_1024_ra.npy'.format("input",data_source,"train"),allow_pickle=True).astype('float32')
    train_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"train"),allow_pickle=True)[:,:].astype('float32')

    train_x, train_y = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(train_Y)
    val_x, val_y = torch.from_numpy(val_X[:,:,:3]), torch.from_numpy(val_Y)
    print("# training samples {}".format(int(len(train_x))))

    train_loop(local_path, dev, train_x, train_y, val_x, val_y)

    # plot #
    plot_loss(local_path)
