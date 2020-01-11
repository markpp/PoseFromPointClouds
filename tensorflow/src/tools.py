import numpy as np
import argparse
import os
import json
import vg
import random
'''
# takes a reference vector and computes azimuth and elevation to align the view
def vec2azel(ref):
    ref_len = np.linalg.norm(ref)
    ref = [ref[0]/ref_len, ref[1]/ref_len, ref[2]/ref_len]
    v1 = np.array(ref)
    v2 = np.array([1,0,0])
    v_d = v1 - v2

    #a = np.arctan(v_d[0]/v_d[1]) # phi
    #e = np.arctan(v_d[2]/v_d[1]) # theta
    #print('azimuth = '+str(a)+', elevation = '+str(e))
    a = vg.signed_angle(np.array(norm),np.array([1,0,0]), look=vg.basis.z, units="rad")
    e = vg.signed_angle(np.array(norm),np.array([1,0,0]), look=vg.basis.y, units="rad")
    #print('azimuth = '+str(a)+', elevation = '+str(e))
'''
# takes a reference vector and computes azimuth and elevation to align the view
def vec2azel(ref, units = "deg"):
    # normalize
    #ref_len = np.linalg.norm(ref)
    #ref = [ref[0]/ref_len, ref[1]/ref_len, ref[2]/ref_len]
    ref = np.array(ref)

    #method 1
    #v_d = ref - np.array([1,0,0])
    #a = np.degrees(np.arctan(v_d[0]/v_d[1])) # phi broken
    #e = np.degrees(np.arctan(v_d[2]/v_d[1])) # theta ok
    #print('azimuth = '+str(a)+', elevation = '+str(e))

    # method 2
    a = vg.signed_angle(ref,vg.basis.x, look=vg.basis.z, units=units)
    e = vg.signed_angle(ref,vg.basis.y, look=vg.basis.x, units=units)
    print('azimuth = '+str(a)+', elevation = '+str(e))
    return -a, -e

def generate_axis_lines(p0, nx, ny, nz, line_len=0.12, n_points = 50):
    p_x_axis = p0+np.outer(np.linspace(0,line_len,n_points),nx)
    c_x_axis = np.full((len(p_x_axis),3),[1, 0, 0])
    p_y_axis = p0+np.outer(np.linspace(0,line_len,n_points),ny)
    c_y_axis = np.full((len(p_y_axis),3),[0, 1, 0])
    p_axis = np.concatenate((p_x_axis, p_y_axis), axis=0)
    c_axis = np.concatenate((c_x_axis, c_y_axis), axis=0)
    p_z_axis = p0+np.outer(np.linspace(0,line_len,n_points),nz)
    c_z_axis = np.full((len(p_z_axis),3),[0, 0, 1])
    p_axis = np.concatenate((p_axis, p_z_axis), axis=0)
    c_axis = np.concatenate((c_axis, c_z_axis), axis=0)
    return p_axis, c_axis

def generate_normal(p0, nx, line_len=0.12, n_points = 50):
    p_norm = p0+np.outer(np.linspace(0,line_len,n_points),nx)
    c_norm = np.full((len(p_norm),3),[1, 0, 0])
    return p_norm[2:], c_norm[2:]

def generate_point(p0, r=0.0015, n_points = 100):
    p_point = []
    for _ in range(n_points):
        p_point.append([p0[0]+random.uniform(-r, r),
                        p0[1]+random.uniform(-r, r),
                        p0[2]+random.uniform(-r, r)])
    c_point = np.full((n_points,3),[0, 1, 0])
    return np.array(p_point), c_point

def load_point_normal(pose_path):
    with open(pose_path, 'r') as data_file:
        json_data = data_file.read()
        jps = json.loads(json_data)
        jp = jps[0]
        p0 = float(jp["pos"]["x"]),float(jp["pos"]["y"]),float(jp["pos"]["z"])
        norm = float(jp["orn"]["x"]),float(jp["orn"]["y"]),float(jp["orn"]["z"])
    return p0, norm

def load_pose(pose_path):
    with open(pose_path) as pose_file:
        jp = json.load(pose_file)
        p0 = float(jp["pos"]["x"]),float(jp["pos"]["y"]),float(jp["pos"]["z"])
        x, y, z, w = float(jp["orn"]["x"]), float(jp["orn"]["y"]), float(jp["orn"]["z"]), float(jp["orn"]["w"])
    return p0, [1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)] # left vector
