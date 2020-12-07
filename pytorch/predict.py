import torch
import json
import argparse
import numpy as np
import os
from itertools import islice
from config import data_source, out_dim

local_path = os.path.join(os.getcwd(), "models")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def print_model_info(model, path):
    file = open(path,'w')
    file.write("Model's parameters:\n")
    file.write("total parms: {}\n".format(sum(p.numel() for p in model.parameters())))
    file.write("trainable parms: {}\n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    file.write("Model's state_dict:\n")
    for param_tensor in model.state_dict():
        file.write("{}, \t {}".format(param_tensor,model.state_dict()[param_tensor].size()))
    file.close()

if __name__ == '__main__':

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data #
    #test_X = np.load('{}/{}_x_1024_ra.npy'.format(data_source,"test"),allow_pickle=True).astype('float32')
    #test_Y = np.load('{}/{}_y.npy'.format(data_source,"test"),allow_pickle=True)[:,:out_dim].astype('float32')
    val_X = np.load('{}/{}_x_1024_ra.npy'.format(data_source,"val"),allow_pickle=True).astype('float32')
    val_Y = np.load('{}/{}_y.npy'.format(data_source,"val"),allow_pickle=True)[:,:out_dim].astype('float32')
    train_X = np.load('{}/{}_x_1024_ra.npy'.format(data_source,"train"),allow_pickle=True).astype('float32')
    train_Y = np.load('{}/{}_y.npy'.format(data_source,"train"),allow_pickle=True)[:,:out_dim].astype('float32')

    with torch.no_grad():
        model_path = os.path.join(local_path,"final_model.pkl")
        if os.path.exists(model_path):
            train_x, val_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3])
            #test_x = torch.from_numpy(test_X[:,:,:3])
            datasets = []
            #datasets.append(["test",test_x])
            datasets.append(["train",train_x])
            datasets.append(["val",val_x])

            model = torch.load("{}/final_model.pkl".format(local_path))
            model.eval()
            model = model.to(dev)

            #print_model_info(model, "{}/model.txt".format(local_path))

            for data in datasets:
                print("{}, x {}".format(data[0],data[1].shape))
                preds = np.empty((0,3+out_dim), float)
                feats = np.empty((0,1024), float)

                for batch_idx in list(chunk(range(len(data[1])), 64)):
                    batch_idx = list(batch_idx)
                    x = data[1][batch_idx].to(dev)
                    pred_p, feat_p, pred_n, feat = model(x)
                    pred = torch.cat([pred_p, pred_n], 1)
                    if dev.type == 'cuda':
                        pred = pred.cpu()
                        feat = feat.cpu()
                    pred = pred.data.numpy()
                    feat = feat.data.numpy()
                    preds = np.append(preds, pred, axis=0)
                    feats = np.append(feats, feat, axis=0)
                print("preds {}".format(preds.shape))
                np.save("{}/{}_pred.npy".format(data_source,data[0]),preds)
                #np.save("{}/{}_feat.npy".format(data_source,data[0]),feats)
        else:
            print("model has not finished training: {}".format(local_path))
