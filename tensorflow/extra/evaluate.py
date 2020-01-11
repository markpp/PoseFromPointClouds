import numpy as np
import vg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
import json


def vect2angle(a,b):
    return np.rad2deg(math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1]))

def plot_dist(d0,d1=None,d2=None,labels=["Test","Val","Train"],name='noname'):

    ax = sns.distplot(d0,kde=False,color="r",bins=25,norm_hist=True,label=labels[0])
    if d2 is not None:
        ax = sns.distplot(d1,kde=False,color="g",bins=25,norm_hist=True,label=labels[1])
    if d2 is not None:
        ax = sns.distplot(d2,kde=False,color="b",bins=25,norm_hist=True,label=labels[2])
    ax.legend()
    ax.set_ylabel('Normalized freqency')
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if 'pos_err' in name:
        ax.set_xlabel('[m]')
        ax.set_title('Difference in positions between ground-truth and prediction')
    elif 'ang_err' in name:
        ax.set_xlabel('[deg]')
        ax.set_title('Difference in normals between ground-truth and prediction')
    else:
        ax.set_xlabel('[]')
        ax.set_title('X compared to ground truth')

    plt.savefig('{}.png'.format(name))
    plt.clf()


def evaluate_point_normals(gts,preds):
    pos, dist, norm, ang = [], [], [], []
    idx = 0
    for gt, pred in zip(gts,preds):
        # numeric differences
        diff = gt-pred
        pos.append(diff[:3])
        dist.append(np.linalg.norm(diff[:3]))
        norm.append(diff[3:])

        # angle difference
        angles = [vg.signed_angle(gt[3:],pred[3:], look=vg.basis.y, units="deg"),
                  vg.signed_angle(gt[3:],pred[3:], look=vg.basis.z, units="deg"),
                  vg.signed_angle(gt[3:],pred[3:], look=vg.basis.x, units="deg"),
                  vg.angle(gt[3:],pred[3:], units="deg")]
        ang.append(angles)
        idx += 1
    return np.array(pos), np.array(dist), np.array(norm), np.array(ang)


def find_n_best_and_worst(gt_val,pred_val):
    val_pos, val_dist, val_norm, val_ang = evaluate_point_normals(gt_val, pred_val)

    # rank best to worst matching samples
    b2w_idx = np.argsort(val_ang[:,3])

    middle_idx = b2w_idx.shape[0] // 2

    print("best")
    print(b2w_idx[:3])
    print(val_ang[b2w_idx[:3]])
    print("median")
    print(b2w_idx[middle_idx-1:middle_idx+2])
    print(val_ang[b2w_idx[middle_idx-1:middle_idx+2]])
    print("worst")
    print(b2w_idx[-3:])
    print(val_ang[b2w_idx[-3:]])

if __name__ == '__main__':
    dataset = 'lay'
    local_path = "input/{}".format(dataset)

    dfs_pos, dfs_ang = [], []

    for data in ["test","val","train"][:]:
        gt = np.load(os.path.join(local_path,'{}_Y.npy'.format(data)))
        pred = np.load(os.path.join(local_path,'pred_{}.npy'.format(data)))

        pos, dist, norm, angs = evaluate_point_normals(gt,pred)
        ang = angs[:,3]
        print("{} GT: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(data,dist.mean(),dist.std(), ang.mean(),ang.std()))

        dist[dist > 0.04] = 0.04
        df_pos = pd.DataFrame(dist,columns=['dp'])
        ang[ang > 20.0] = 20.0
        df_ang = pd.DataFrame(ang,columns=['da'])

        dfs_pos.append(df_pos)
        dfs_ang.append(df_ang)

    plot_dist(dfs_pos[0], dfs_pos[1], dfs_pos[2],labels=["test","val","train"],name='plots/pos_err')
    plot_dist(dfs_ang[0], dfs_ang[1], dfs_ang[2],labels=["test","val","train"],name='plots/ang_err')
