import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh
import math

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros(pose.shape)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0];
    new_pose[1] = np.round(pose[1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
# Computes the probability from log-odds form
def logodds2prob(l):
    return 1-(1/(1+np.exp(l)))

# Computes the log-odds from probability
def prob2logodds(p):
    return np.log(p / (1 - p)) 
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    line = bresenham(cell[0], cell[1], endpoint[0], endpoint[1])
    p_values = [0] * len(line)
    for i in range(len(line) - 1):
        p_values[i] = prob_free
    
    p_values[-1] = prob_occ
    inv_sensor_model = np.zeros((len(line),3))
    for i, l in enumerate(line):
        inv_sensor_model[i] = [*l, p_values[i]]

    return inv_sensor_model

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    # Known poses 
    poses = poses2cells(poses_raw, occ_gridmap, map_res)
    
    # Convert gridmap probabilities to logodds
    occ_gridmap = prob2logodds(occ_gridmap)

    i = 0
    # For every pose and laser scan
    for pose, poses_raw, range_raw in zip(poses, poses_raw, ranges_raw):
        range_cell = ranges2cells(range_raw, poses_raw, occ_gridmap, map_res).transpose()

        # Compute and update the probability within the senor range. 
        # Go through all scans in 360 degrees. 
        for endpoint in range_cell:
            ism_array = inv_sensor_model(pose, endpoint, prob_occ, prob_free)
           
            """ Old slow way. """
            # for ism in ism_array:
            #     x = ism[0][0]
            #     y = ism[0][1]
            #     inv_sensor_prob = ism[1]

            #     # Update the grid map by converting probiblity output from the sensor to logvalue and add it to grid value.
            #     occ_gridmap[x][y] = prob2logodds(inv_sensor_prob) + occ_gridmap[x][y] - prob2logodds(prior)
            #print(ism_array)

            """ Faster. Increases performance by ~20 % """
            x = ism_array[:,0]
            y = ism_array[:,1]
            x = x.astype('int')
            y = y.astype('int')

            inv_sensor_prob = ism_array[:,2]
            occ_gridmap[x,y] = prob2logodds(inv_sensor_prob) + occ_gridmap[x,y] - prob2logodds(prior)


        # Just want to count them loopss
        i += 1
        #f i > 100: break
        if (i % 10) == 0: print(f"Pose: {i}")
        
    # Convert back to probabiltiy
    occ_gridmap = logodds2prob(occ_gridmap)

    return occ_gridmap


