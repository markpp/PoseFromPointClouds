import numpy as np
import argparse
import os
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D
import vg

import sys
sys.path.append('..')
from src.in_out import sample_x
from config import data_source

# takes a reference vector and computes azimuth and elevation to align the view
def vec2azel(ref, type='y', units = "deg"):
    ref = np.array(ref)

    if type == 'y':
        a = vg.signed_angle(ref,vg.basis.x, look=vg.basis.z, units=units)
        e = vg.signed_angle(ref,vg.basis.y, look=vg.basis.x, units=units)
    elif type == 'z':
        a = vg.signed_angle(ref,vg.basis.x, look=vg.basis.y, units=units)
        e = vg.signed_angle(ref,vg.basis.z, look=vg.basis.x, units=units)
    else:
        a = vg.signed_angle(ref,vg.basis.z, look=vg.basis.y, units=units)
        e = vg.signed_angle(ref,vg.basis.x, look=vg.basis.z, units=units)
    print('azimuth = '+str(a)+', elevation = '+str(e))
    return -a, -e

def evaluate_point_normals(gts,preds):
    pos, dist, norm, ang = [], [], [], []
    idx = 0
    for gt, pred in zip(gts,preds):
        # numeric differences
        diff = gt-pred
        pos.append(diff[:3])
        dist.append(np.linalg.norm(diff[:3]))
        norm.append(diff[3:6])

        # angle difference
        angles = [vg.signed_angle(gt[3:6],pred[3:6], look=vg.basis.y, units="deg"),
                  vg.signed_angle(gt[3:6],pred[3:6], look=vg.basis.z, units="deg"),
                  vg.signed_angle(gt[3:6],pred[3:6], look=vg.basis.x, units="deg"),
                  vg.angle(gt[3:6],pred[3:6], units="deg")]
        ang.append(angles)
        idx += 1
    return np.array(pos), np.array(dist), np.array(norm), np.array(ang)[:,3]

def find_n_best_med_worst(measures,n=4):
    # rank best to worst matching samples
    b2w_idx = np.argsort(measures)
    print("best")
    print(b2w_idx[:n])
    print(measures[b2w_idx[:n]])
    print("median")
    middle_idx = b2w_idx.shape[0] // 2
    offset = n//2
    print(b2w_idx[middle_idx-offset:middle_idx+offset])
    print(measures[b2w_idx[middle_idx-offset:middle_idx+offset]])
    print("worst")
    print(b2w_idx[-n:])
    print(measures[b2w_idx[-n:]])
    return np.concatenate((b2w_idx[:n],b2w_idx[middle_idx-offset:middle_idx+offset],b2w_idx[-n:]),axis=0)

if __name__ == '__main__':

    dataset = "val"
    gt = np.load('../{}/{}_y.npy'.format(data_source,dataset))[:,:6]
    pred = np.load('../{}/{}_pred.npy'.format(data_source,dataset))[:,:6]
    x = np.load('../{}/{}_x_1024_ra.npy'.format(data_source,dataset),allow_pickle=True)
    #x = np.load('../input/{}/test_X.npy'.format(dataset),allow_pickle=True)
    #x = sample_x(x, 1024)

    pos, dist, norm, ang = evaluate_point_normals(gt, pred)

    idx_list = find_n_best_med_worst(ang)
    print(idx_list)

    for i,idx in enumerate(idx_list):
        print("idx: {}".format(idx))
        y = gt[idx]
        print("gt: {}".format(y))
        y_ = pred[idx]
        print("pred: {}".format(y_))
        points = x[idx]
        print("x: {}".format(points.shape))

        point_colors = np.full((points.shape[0],3),[1, 0.7, 0.75]) # fixed color

        p0, nx = y[:3], y[3:6]
        #nz = np.cross(nx,[0,1,0])
        #ny = np.cross(nz,nx)
        #ny = y[6:9]
        #nz = np.cross(ny,nx)

        p0_, nx_ = y_[:3], y_[3:6]
        #nz_ = np.cross(nx_,[0,1,0])
        #ny_ = np.cross(nz_,nx_)
        #ny_ = y_[6:9]
        #nz_ = np.cross(ny_,nx_)

        # Create the figure
        fig = plt.figure(figsize=(6,6))

        # Add an axes
        ax = fig.add_subplot(111,projection='3d')
        ax.set_title('Position error {:.3f}m, normal error {:.2f}°'.format(dist[idx],ang[idx]))

        # and plot the point
        ax.scatter(points[:,0] , points[:,1] , points[:,2],  color=point_colors, alpha=0.5)
        #ax.scatter(points[:,0] , points[:,1] , points[:,2],  color='pink', marker='o')
        #if p0 is not None and nx is not None:
            #p_axis, c_axis = generate_axis_lines(p0, nx, ny, nz)
            #p_pose, c_pose = generate_arrow(p0, nx)
            #ax.scatter(p_pose[:,0] , p_pose[:,1] , p_pose[:,2], color=c_pose, alpha=0.5)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_axis_off()

        lim_val = 0.10
        ax.set_xlim3d(-lim_val, lim_val)
        ax.set_ylim3d(-lim_val, lim_val)
        ax.set_zlim3d(-lim_val, lim_val)

        if nx is not None:
            #a, e = vec2azel(ny, type='y', units="deg")
            #ax.view_init(elev=e, azim=a)

            ax.quiver(*p0, *nx, length=0.15, normalize=False, color='b')
            ax.quiver(*p0_, *nx_, length=0.1, normalize=False, color='r')
            #ax.quiver(*p0_, *ny_, length=0.1, normalize=False, color='r')
            #ax.quiver(*p0_, *nz_, length=0.1, normalize=False, color='r')

            plt.savefig("../plots/{}_{}.png".format(i,idx), bbox_inches='tight')
        plt.show()
