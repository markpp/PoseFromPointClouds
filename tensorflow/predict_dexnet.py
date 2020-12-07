import os
import numpy as np
import random
import json
from tensorflow.keras.models import load_model
from config import data_source, out_dim, batch_size, n_points
from pyntcloud import PyntCloud


class CameraIntrinsics(object):
    """A set of intrinsic parameters for a camera. This class is used to project
    and deproject points.
    """

    def __init__(self, frame, fx, fy=None, cx=0.0, cy=0.0, skew=0.0, height=None, width=None):
        """Initialize a CameraIntrinsics model.
        Parameters
        ----------
        frame : :obj:`str`
            The frame of reference for the point cloud.
        fx : float
            The x-axis focal length of the camera in pixels.
        fy : float
            The y-axis focal length of the camera in pixels.
        cx : float
            The x-axis optical center of the camera in pixels.
        cy : float
            The y-axis optical center of the camera in pixels.
        skew : float
            The skew of the camera in pixels.
        height : float
            The height of the camera image in pixels.
        width : float
            The width of the camera image in pixels
        """
        self._frame = frame
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._skew = float(skew)
        self._height = int(height)
        self._width = int(width)

        # set focal, camera center automatically if under specified
        if fy is None:
            self._fy = fx

        # set camera projection matrix
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    def project(self, point, round_px=True):
        """Projects a point onto the camera image plane.
        """
        points_proj = self._K.dot(point)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, np.newaxis]
        point_depths = np.tile(points_proj[2,:], [3, 1])
        points_proj = np.divide(points_proj, point_depths)
        if round_px:
            points_proj = np.round(points_proj)
       
        return points_proj

    def deproject_pixel(self, depth, pixel):
        """Deprojects a single pixel with a given depth into a 3D point.
        Parameters
        ----------
        depth : float
            The depth value at the given pixel location.
        pixel : :obj:`autolab_core.Point`
            A 2D point representing the pixel's location in the camera image.
        Returns
        -------

        """
        point_3d = depth * np.linalg.inv(self._K).dot(np.r_[pixel, 1.0])
        return point_3d

    def deproject(self, depth_image):
        """Deprojects a DepthImage into a PointCloud.
        Parameters
        ----------
        depth_image : :obj:`DepthImage`
            The 2D depth image to projet into a point cloud.
        Returns
        -------
        :obj:`autolab_core.PointCloud`
            A 3D point cloud created from the depth image.
        Raises
        ------
        ValueError
            If depth_image is not a valid DepthImage in the same reference frame
            as the camera.
        """
        # create homogeneous pixels
        row_indices = np.arange(depth_image.shape[0])
        col_indices = np.arange(depth_image.shape[1])
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3,1])

        # deproject
        points_3d = depth_arr * np.linalg.inv(self._K).dot(pixels_homog)
        points_3d = np.swapaxes(points_3d,0,1)
        return points_3d


def pose(camera_intr, angle):
    """Computes the 3D pose of the grasp relative to the camera.
    Returns
    -------

    """
    # Compute 3D grasp axis in camera basis.
    grasp_axis_im = np.array([np.cos(angle), np.sin(angle)])
    grasp_axis_im = grasp_axis_im / np.linalg.norm(grasp_axis_im)
    grasp_axis_camera = np.array([grasp_axis_im[0], grasp_axis_im[1], 0])
    grasp_axis_camera = grasp_axis_camera / np.linalg.norm(grasp_axis_camera)

    # Convert to 3D pose.
    grasp_rot_camera, _, _ = np.linalg.svd(grasp_axis_camera.reshape(3, 1))
    #grasp_x_camera = grasp_approach_dir
    #if grasp_approach_dir is None:
    grasp_x_camera = np.array([0, 0, 1])  # Align with camera Z axis.
    grasp_y_camera = grasp_axis_camera
    grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
    grasp_z_camera = grasp_z_camera / np.linalg.norm(grasp_z_camera)
    grasp_y_camera = np.cross(grasp_z_camera, grasp_x_camera)
    grasp_rot_camera = np.array([grasp_x_camera, grasp_y_camera, grasp_z_camera])

    if np.linalg.det(grasp_rot_camera) < 0:  # Fix reflections due to SVD.
        grasp_rot_camera[:, 0] = -grasp_rot_camera[:, 0]

    return grasp_rot_camera[2]

def compute_angle_point(x,y,angle,length=5):
    x2 =  int(x + length * math.cos(angle))
    y2 =  int(y + length * math.sin(angle))
    return (x2, y2)

def move_to_origo(points, center_mass=True):
    # find center
    if center_mass:
        xmean = np.mean(points[:,0])
        ymean = np.mean(points[:,1])
        zmean = np.mean(points[:,2])
        center = [xmean,ymean,zmean]
    else:
        xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
        ymin, ymax = np.min(points[:,1]), np.max(points[:,1])
        zmin, zmax = np.min(points[:,2]), np.max(points[:,2])
        center = [xmin+(xmax-xmin)/2,ymin+(ymax-ymin)/2,zmin+(zmax-zmin)/2]
    # translate points to 0,0,0
    points[:,0] -= center[0]
    points[:,1] -= center[1]
    points[:,2] -= center[2]
    return points, center

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

if __name__ == '__main__':
    in_path = '{}/grasps/tensors'.format(data_source)

    model = load_model("models/dex/model_pn.h5", compile = False)

    im_preds = []
    gts, preds, X = [], [], []

    predict = True

    for idx in range(1557)[:]:
        idx = str(idx).zfill(5)

        grasp_metrics = np.load('{}/grasp_metrics_{}.npz'.format(in_path,idx))['arr_0']
        grasped_obj_keys = np.load('{}/grasped_obj_keys_{}.npz'.format(in_path,idx))['arr_0']
        splits = np.load('{}/split_{}.npz'.format(in_path,idx))['arr_0']
        grasps = np.load('{}/grasps_{}.npz'.format(in_path,idx))['arr_0']
        camera_intrs = np.load('{}/camera_intrs_{}.npz'.format(in_path,idx))['arr_0']
        tf_depth_ims = np.load('{}/tf_depth_ims_{}.npz'.format(in_path,idx))['arr_0']

        for i in range(100):
            if grasp_metrics[i] < 0.5:
                continue
            object_class = grasped_obj_keys[i].decode("utf-8").split('~')[-1]
            print("object type: {}".format(object_class))
            out_path = '{}/output/{}'.format(data_source,object_class)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            camera_intr = CameraIntrinsics("frame_0",*camera_intrs[i])

            # create and save point cloud
            points = camera_intr.deproject(tf_depth_ims[i])
            X.append(points)
            #cloud = PyntCloud(pd.DataFrame(points,columns=['x', 'y', 'z']))
            #cloud.to_file(os.path.join(out_path,"{}_{}.ply".format(idx,i)))

            #print("grasp: {}".format(grasps[i]))
            #im_x, im_y = int(grasps[i][0]), int(grasps[i][1])
            im_x, im_y = 96//2, 96//2
            #depth_data = depth_data.astype(np.float32) / 1000.0
            p0 = camera_intr.deproject_pixel(tf_depth_ims[i,im_x,im_y], [im_x,im_y])            
            norm = pose(camera_intr, grasps[i][3])
            norm = norm / np.linalg.norm(norm)

            gts.append([p0[0],p0[1],p0[2],norm[0],norm[1],norm[2]])
            
            with open(os.path.join(out_path,"{}_{}.json".format(idx,i)), 'w') as f:
                jo = [{
                    "pos": {
                        "x": p0[0],
                        "y": p0[1],
                        "z": p0[2]
                    },
                    "nx": {
                        "x": norm[0],
                        "y": norm[1],
                        "z": norm[2]
        
                    }
                }]
                f.write(json.dumps(jo, indent=2, sort_keys=False))

            '''
            # save depth img
            #print(tf_depth_ims[i])
            depth_img = (tf_depth_ims[i]*255).astype('uint8')
            cv2.imwrite(os.path.join(out_path,"{}_{}.png".format(idx,i)), depth_img)

            gt = cv2.circle(depth_img.clone(),(im_x, im_y), 2, 255, 1)
            cv2.line(gt,compute_angle_point(im_x, im_y, grasps[i][3], length=-5),compute_angle_point(im_x, im_y, grasps[i][3]),255,1)
            cv2.imwrite(os.path.join(out_path,"{}_{}_gt.png".format(idx,i)), cv2.resize(gt,None,fx=4.0,fy=4.0))
            '''

            x, offset = move_to_origo(sample_N_random(points))
            pred = model.predict(np.expand_dims(x, axis=0))
            pred = np.concatenate((pred[0][0], pred[1][0]), axis=0)

            t, nx = (pred[:3]+offset), pred[3:6]
            nx = nx / np.linalg.norm(nx)
            
            preds.append([t[0],t[1],t[2],nx[0],nx[1],nx[2]])

            with open(os.path.join(out_path,"{}_{}_.json".format(idx,i)), 'w') as f:
                jo = [{
                    "pos": {
                        "x": float(t[0]),
                        "y": float(t[1]),
                        "z": float(t[2])
                    },
                    "nx": {
                        "x": float(nx[0]),
                        "y": float(nx[1]),
                        "z": float(nx[2])
        
                    }
                }]
                f.write(json.dumps(jo, indent=2, sort_keys=False))

            proj = camera_intr.project(t)
            #print(proj[0])
            #print(proj[1])
            preds.append([proj[0],proj[1]])
            '''
            pr = cv2.circle(depth_img.clone(),(im_x, im_y), 2, 255, 1)
            cv2.line(gt,compute_angle_point(im_x, im_y, grasps[i][3], length=-5),compute_angle_point(im_x, im_y, grasps[i][3]),255,1)
            cv2.imwrite(os.path.join(out_path,"{}_{}_pr.png".format(idx,i)), cv2.resize(pr,None,fx=4.0,fy=4.0))
            '''
    np.save(os.path.join(out_path,"im_preds.npy"),np.array(im_preds))
    np.save(os.path.join(out_path,"test_Y.npy"),np.array(gts))
    np.save(os.path.join(out_path,"test_pred.npy"),np.array(preds))
    np.save(os.path.join(out_path,"test_X.npy"),np.array(X))

        