import os
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle
from tqdm import tqdm
from numba import jit, prange

import pr2_utils as utils


def show_image() -> None:
    """
    Show Vehicle dimensions and coordinates
    """
    vehicle_img = plt.imread("data/vehicle_cofig.png")
    plt.imshow(vehicle_img)
    plt.show()


def show_lidar(angles: np.ndarray, ranges: np.ndarray, title="Lidar Scan Data") -> None:
    """
    Show lidar in polar coordinates
    """
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(80)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title(title, va='bottom')
    plt.show()


def show_laserXY(xs, ys) -> None:
    """
    plot lidar points in cartesian coordinate
    """
    # fig1 = plt.figure()
    plt.plot(xs, ys, '.k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def show_map(map, title='') -> None:
    """
    plot Occupancy grid map
    :param map: grid map
    :type: numpy array
    :param title: title of plot
    :type: str
    """
    plt.figure()
    plt.imshow(map, cmap="hot")
    plt.title(title)
    # plt.grid(True)
    plt.show()


@jit(nopython=True)
def cartesian2polar(x, y):
    """
    convert from cartesian coordinates to polar coordinates
    :param x
    :type: float or numpy array
    :param y
    :type: float or numpy array
    :return r:ranges in meters
            theta: angles in radians
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


@jit(nopython=True)
def polar2cartesian(r, theta):
    """
    convert from polar coordinates to cartesian coordinates
    :param theta angle in radian
    :type: float or numpy array
    :param r range
    :type: float or numpy array
    :return x, y in cartesian coordinates
    """
    a = r * np.exp(1j * theta)
    return a.real, a.imag


def polar2xyz(ranges: np.ndarray, lidar_param: dict, verbose=False) -> np.ndarray:
    """
    Convert from polar coordinates to cartesian coordinate with z axis fill with zeros
    Remove scan points that are too close or too far,
    Only consider points between [min_range=2, max_range=80]
    * Measurements between 2m-75m are recommended to be included as valid data.
    :param: lidar_data (one slice of observations)
    :type:  numpy.array
    :param: index
    :type: int
    :param: min_range=2
    :type: int
    :param: max_range=80
    :type: int
    :return: n * 1-D vector [x, y, z=0].T
        where  n = 286, -> FOV: 190 (degree)/Angular resolution: 0.666 (degree) = 285.28
        if all cloud points are valid
    """
    # assert isinstance(ranges, np.ndarray)
    # assert isinstance(lidar_param, dict)
    max_range = lidar_param["max_range"]
    min_range = lidar_param["min_range"]
    # angles = np.deg2rad(np.linspace(-5, 185, 286))
    angles = lidar_param["angles"]
    # if verbose:
    #     show_lidar(angles, ranges, title="Raw Lidar Scan Data")
    # Filter out noisy data (r<2 and r>80)
    indValid = np.logical_and((ranges <= max_range), (ranges >= min_range))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # if verbose:
    #     show_lidar(angles, ranges, title="Valid Lidar Scan Data")
    # Convert from polar to cartesian coordinates
    x, y = polar2cartesian(ranges, angles)
    # sanity check car2pol conversion
    r, theta = cartesian2polar(x, y)
    assert np.allclose(r, ranges)
    # attach dummy z-axis
    z = np.zeros((1, len(ranges)))
    return np.vstack((x, y, z))


def get_lidar_param(verbose=False) -> dict:
    """
    FOV: 190 (degree), Start angle: -5 (degree), End angle: 185 (degree),
    Angular resolution: 0.666 (degree)
    Max range: 80 (meter)
    * LiDAR rays with value 0.0 represent infinite range observations.

    Lidar sensor (LMS511) extrinsic calibration parameter from vehicle
    RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
    RPY: 142.759 0.0584636 89.9254
    R: 0.00130201 0.796097 0.605167, 0.999999 -0.000419027 -0.00160026, -0.00102038 0.605169 -0.796097
    T: 0.8349 -0.0126869 1.76416
    :return: lidar_param
    """
    R_deg, P_deg, Y_deg = (142.759, 0.0584636, 89.9254)
    RPY = namedtuple("RPY_angle", ["roll_angle", "pitch_angle", "yaw_angle"])
    RPY_deg = RPY(R_deg, P_deg, Y_deg)
    RPY_rad = np.deg2rad(RPY(R_deg, P_deg, Y_deg))

    V_R_L = np.array([[0.00130201, 0.796097, 0.605167],
                      [0.999999, -0.000419027, -0.00160026],
                      [-0.00102038, 0.605169, -0.796097]
                      ],
                     dtype=np.float64)
    # position [x,y,z].T denoted as V_p_L
    V_p_L = np.array([0.8349, -0.0126869, 1.76416], dtype=np.float64)

    # verify R
    Rot = get_R(*RPY_rad)
    if verbose:
        print(f"R: {Rot}\nRot:{Rot}")
        print(f"Diff(R-Rot):{np.subtract(V_R_L, Rot)}\n")
        print(f"Translation p: {V_p_L}")
    assert np.allclose(V_R_L, Rot)

    lidar_param = dict()
    lidar_param["R_deg"] = R_deg
    lidar_param["P_deg"] = P_deg
    lidar_param["Y_deg"] = Y_deg
    lidar_param["RPY_deg"] = RPY_deg
    lidar_param["RPY_rad"] = RPY_rad
    lidar_param["V_Rot_L"] = V_R_L
    lidar_param["V_pos_L"] = V_p_L
    lidar_param["V_T_L"] = get_T(V_R_L, V_p_L)
    lidar_param["L_T_V"] = np.linalg.inv(lidar_param["V_T_L"])
    lidar_param["FOV"] = 190
    lidar_param["start_angle"] = -5
    lidar_param["end_angle"] = 185
    lidar_param["angles"] = np.deg2rad(np.linspace(-5, 185, 286, dtype=np.float64))
    lidar_param["max_range"] = 80
    lidar_param["min_range"] = 2
    lidar_param["angular_resolution"] = 0.666
    info = """FOV: 190 (degree), Start angle: -5 (degree), End angle: 185 (degree),
    Angular resolution: 0.666 (degree)
    Max range: 80 (meter)
    * LiDAR rays with value 0.0 represent infinite range observations."""
    lidar_param["info"] = info
    if verbose:
        print(f"lidar_param_INFO: {info}")
    return lidar_param


def get_FOG_param(verbose=False) -> dict:
    """
    FOG (Fiber Optic Gyro) extrinsic calibration parameter from vehicle
    RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix, meter)
    RPY: [0. 0. 0.]
    R: [1 0 0, 0 1 0, 0 0 1]
    T: [-0.335, -0.035, 0.78]
    * The sensor measurements are stored as [timestamp, delta roll, delta pitch, delta yaw] in radians.
    :return: FOG_param
    """
    R_deg, P_deg, Y_deg = (0.0, 0.0, 0.0)
    RPY = namedtuple("RPY_angle", ["roll_angle", "pitch_angle", "yaw_angle"])
    RPY_deg = RPY(R_deg, P_deg, Y_deg)
    RPY_rad = np.deg2rad(RPY(R_deg, P_deg, Y_deg))

    V_R_F = np.eye(3, dtype=np.float64)
    # position [x,y,z].T denoted as V_p_F
    V_p_F = np.array([-0.335, -0.035, 0.78], dtype=np.float64)

    FOG_param = dict()
    FOG_param["R_deg"] = R_deg
    FOG_param["P_deg"] = P_deg
    FOG_param["Y_deg"] = Y_deg
    FOG_param["RPY_deg"] = RPY_deg
    FOG_param["RPY_rad"] = RPY_rad
    FOG_param["V_Rot_F"] = V_R_F
    FOG_param["V_pos_F"] = V_p_F
    FOG_param["V_T_F"] = get_T(V_R_F, V_p_F)
    FOG_param["F_T_V"] = np.linalg.inv(FOG_param["V_T_F"])
    info = "* FOG measurements are stored as [timestamp, delta roll, delta pitch, delta yaw] in " \
           "radians. "
    FOG_param["info"] = info
    if verbose:
        print(f"FOG_param_INFO: {info}")
    return FOG_param


def get_encoder_param(verbose=False) -> dict:
    """
    Encoder calibrated parameter
    Encoder resolution: 4096
    Encoder left wheel diameter: 0.623479
    Encoder right wheel diameter: 0.622806
    Encoder wheel base: 1.52439

    * The encoder data is stored as [timestamp, left count, right count].
    :return: encoder_param
    """
    encoder_param = dict()
    encoder_param["ticks_per_revolution"] = 4096
    encoder_param["left_diameter"] = 0.623479
    encoder_param["right_diameter"] = 0.622806
    encoder_param["base"] = 1.52439
    info = "* The encoder data is stored as [timestamp, left count, right count]."
    encoder_param["info"] = info
    if verbose:
        print(f"encoder_param_INFO:{info}")
    return encoder_param


def get_left_cam_param() -> dict:
    """
    Left camera parameter
    Stereo camera (based on left camera) extrinsic calibration parameter from vehicle
    RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
    RPY: -90.878 0.0132 -90.3899
    R: -0.00680499 -0.0153215 0.99985 -0.999977 0.000334627 -0.00680066 -0.000230383 -0.999883 -0.0153234
    T: 1.64239 0.247401 1.58411
    :return: left_cam_param
    """
    left_cam_param = dict()
    left_cam_param["img_width"] = 1280
    left_cam_param["img_height"] = 560
    left_cam_param["cam_name"] = "/stereo/left"
    left_cam_param["projection_matrix"] = np.array([7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.,
                                                    0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.,
                                                    0., 0., 1., 0.],
                                                   dtype=np.float64).reshape(3, 4)
    # left_cam_param["intrinsic"] = left_cam_param["projection_matrix"][0:3, 0:3]

    left_cam_param["baseline"] = 475.143600050775 / 1000

    left_cam_param["V_R_C"]= np.array([[-0.00680499, - 0.0153215, 0.99985],
                                       [- 0.999977, 0.000334627, - 0.00680066],
                                       [- 0.000230383, - 0.999883, - 0.0153234]],
                                      dtype=np.float64
                                      )
    left_cam_param["V_P_C"] = np.array([1.64239, 0.247401, 1.58411], dtype=np.float64)
    left_cam_param["V_T_C"] = get_T(left_cam_param["V_R_C"], left_cam_param["V_P_C"])

    return left_cam_param


def get_right_cam_param() -> dict:
    """
    Right camera parameter
    :return: right_cam_param
    """
    #     [fx'  0  cx' Tx]
    # P = [ 0  fy' cy' Ty]
    #     [ 0   0   1   0]
    right_cam_param = dict()
    right_cam_param["img_width"] = 1280
    right_cam_param["img_height"] = 560
    right_cam_param["cam_name"] = "/stereo/right"
    right_cam_param["projection_matrix"] = np.array([7.7537235550066748e+02, 0., 6.1947309112548828e+02, -3.6841758740842312e+02,
                                                     0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.,
                                                     0., 0., 1., 0.],
                                                    dtype=np.float64).reshape(3, 4)
    # Baseline of stereo cameras: 475.143600050775 (mm)
    fx = 7.7537235550066748e+02
    B =475.143600050775/1000
    right_cam_param["baseline"] = B

    Tx = -fx * B
    assert abs(Tx-right_cam_param["projection_matrix"][0,3])<1
    return right_cam_param


def get_R(x: float, y: float, z: float) -> np.ndarray:
    """
    Calculate 3x3 rotation matrix in Euler Angle Parametrization.
    x,y,z must be in radians
    :param x: roll angle
    :type : float
    :param y: pitch angle
    :type : float
    :param z: yaw angle
    :type : float
    return rotation matrix
    """
    R_z = np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]],
                   dtype=np.float64)
    R_y = np.array([[np.cos(y), 0, np.sin(y)],
                    [0, 1, 0],
                    [-np.sin(y), 0, np.cos(y)]],
                   dtype=np.float64)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]],
                   dtype=np.float64)
    # print(R_z @ R_y @ R_x) # equiv to following
    return np.dot(np.dot(R_z, R_y), R_x)


def get_T(Rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Calculate Rigid Body Pose
    :param: R: rotation matrix
    :type: 3x3 numpy array
    :param: p: translation matrix
    :type: (3,) numpy array
    :return: pose T [R P
                     0.T 1 ]
    """
    assert isinstance(Rot, np.ndarray)
    assert isinstance(pos, np.ndarray)
    assert np.size(Rot) == 9
    assert pos.ndim == 1
    x, y, z = pos
    T = np.array([[0, 0, 0, x],
                  [0, 0, 0, y],
                  [0, 0, 0, z],
                  [0, 0, 0, 1]])
    T[0:3, 0:3] = Rot
    return T


@jit(nopython=True)
def reg2homo(X: np.ndarray) -> np.ndarray:
    """
    Convert Matrix to homogenous coordinate
    :param X: matrix/vector
    :type :numpy array
    return X_ -> [[X]
                  [1]]
    """
    # assert isinstance(X, np.ndarray)
    ones = np.ones((1, X.shape[1]), dtype=np.float64)
    # print(ones)
    X_ = np.concatenate((X, ones), axis=0)
    return X_


@jit(nopython=True)
def lidar2body(s_L_: np.ndarray, V_T_L: np.ndarray, F_T_V: np.ndarray, verbose=False) -> np.ndarray:
    """
    Convert point clouds from Lidar frame to Body frame(FOG frame)
    :param s_L_:  point clouds in homogenous coordinate
    :param V_T_L: pose from lidar to vehicle
    :param F_T_V: pose from vehicle to FOG
    :param verbose:
    :return: s_B_
    """
    # assert isinstance(s_L_, np.ndarray) and s_L_.shape[0] == 4
    # assert isinstance(V_T_L, np.ndarray) and V_T_L.shape[0] == 4 and V_T_L.shape[1] == 4
    # assert isinstance(F_T_V, np.ndarray) and F_T_V.shape[0] == 4 and F_T_V.shape[1] == 4
    # Transform from lidar frame to vehicle frame s_L -> s_V
    s_V0_ = V_T_L @ s_L_
    # Define FOG frame to be the body frame, coincide with vehicle frame
    s_F0_ = F_T_V @ s_V0_  # s_V -> s_F
    # if verbose:
    #     print(f"s_V[0](homogenous): {s_V0_.shape}")
    #     print(f"s_F[0](homogenous): {s_F0_.shape}")
    return s_F0_


def differential_drive_model(Xt, vt, delta_theta, dt) -> np.ndarray:
    """
    Discrete-time Differential-drive Kinematic Model
    :param Xt: state   [x, y, theta].T (x_pos, y_pos, yaw_angle_rad)
    :param vt: control [vt, wt]      (linearVelocity, angularVelocity)
    :param delta_theta: yaw angle change
    :param dt: time duration
    :return: xt+1, yt+1, theta_t+1
    """
    theta_t = Xt[2, 0]
    dv = np.array([[vt * np.cos(theta_t)],
                   [vt * np.sin(theta_t)],
                   ],
                  dtype=np.float64)
    dx = np.vstack((dt * dv, delta_theta))
    return Xt + dx


def dead_reckoning(path, expert=False, verbose=False) -> np.ndarray:
    """
    Perform dead_reckoning
    :param expert: load dead reckoning trajectory
    :param path: path to sync_fog_encoder_df
    :type: str
    :param verbose: bool
    return trajectory
    """
    if not expert:
        sync_fog_encoder_df = pd.read_csv(path)
        total_duration = (max(sync_fog_encoder_df['timestamp']) - min(sync_fog_encoder_df['timestamp'])) * 1e-3
        print(f"total_duration:{total_duration} secs")
        assert abs(sum(sync_fog_encoder_df['dt']) - total_duration) < 1
        num_state = sync_fog_encoder_df.shape[0]
        vt = np.array(sync_fog_encoder_df['linear_velocity(m/s)'], dtype=np.float64)
        delta_yaw = np.array(sync_fog_encoder_df['delta_yaw'], dtype=np.float64)
        tau = np.array(sync_fog_encoder_df['dt'], dtype=np.float64)
        if verbose:
            print(f"sync_fog_encoder_df: {sync_fog_encoder_df.shape}")
            print(sync_fog_encoder_df.head(5))
        # Assume initial state Xt at t=0 = (x:0,y:0,theta:0)
        X0 = np.zeros((3, 1))
        lst = [X0]
        for t in range(num_state):
            X0 = differential_drive_model(X0, vt[t], delta_yaw[t], tau[t])
            lst.append(X0)
        trajectory = np.hstack(lst)
        # with open('dead_reckon_traj.npy', 'wb') as f:
        #     np.save(f, trajectory)
    else:
        with open('dead_reckon_traj.npy', 'rb') as f:
            trajectory = np.load(f)

    plt.figure()
    plt.scatter(trajectory[0, :], trajectory[1, :])
    plt.title("Dead Reckoning (No Noise)")
    plt.show()
    return trajectory


def init_map() -> dict:
    """
    Init MAP
    :return: MAP
    """
    # (x_min=0.0, x_max1238), (y_min=-1012, y_max=0.0)
    MAP = dict()
    MAP['res'] = 1  # meters
    MAP['xmin'] = -100
    MAP['xmax'] = 1350
    MAP['ymin'] = -1350
    MAP['ymax'] = 100
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
    MAP['log_odds_map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64)
    MAP['cell_trajs_map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)

    print(f"map resolution: {MAP['res']}")
    print(f"map size: {MAP['map'].shape}")
    return MAP


def update_map(MAP: dict, Xt: np.ndarray, s_F_: np.ndarray, verbose=False) -> dict:
    """
    Update Probabilistic Occupancy Grid Mapping
    :param MAP:
    :type:dict
    :param Xt: Location of robot in world frame [x,y,theta]
    :type: numpy array
    :param s_F_: lidar scan in body frame in homogenous coordinate
    :type: numpy array
    :param verbose:
    :return: updated MAP
    """
    x_W = Xt[0]
    y_W = Xt[1]
    theta_W = Xt[2]

    # Rotation around z-axis
    W_T_F = np.array(
        [[np.cos(theta_W), -np.sin(theta_W), 0, x_W],
         [np.sin(theta_W), np.cos(theta_W), 0, y_W],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64
    )
    # Body frame and the world frame are perfectly aligned at t=0 -> s_W = I@s_B + 0
    # Lidar data in world frame
    s_W_ = W_T_F @ s_F_

    # Convert homogenous coord to xyz
    # s_W_ = np.delete(s_W_, 3, axis=0)

    ######################################################################################
    # Lidar data in world frame
    ex_W = s_W_[0, :]
    ey_W = s_W_[1, :]

    # convert from meters to cells
    # start point of laser beam in world frame to grid cell
    sx = int(np.ceil((x_W - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
    sy = int(np.ceil((y_W - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
    # cell_traj = np.vstack((sx, sy))
    # end point of laser beam in world frame to grid cell
    ex = np.ceil((ex_W - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    ey = np.ceil((ey_W - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    if verbose:
        print(f"Robot Location in world frame: {Xt}")
        print(f"s_W[0] (homogenous): {s_W_.shape}")
        # print(f"s_W[0]  with z-axis: {s_W_.shape}")
        print(f"({sx},{sy})")

    # global rays
    # rays = []
    for i in range(ex.shape[0]):
        # Bresenham2D() assumes sx, sy, ex, ey are already in "cell coordinates"
        ray = utils.bresenham2D(sx=sx, sy=sy, ex=ex[i], ey=ey[i])
        # rays.append(ray)
        xis = ray[0, :].astype(np.int16)
        yis = ray[1, :].astype(np.int16)

        # log-odds if zt indicates mi is occupied +log4, -log4 otherwise
        MAP['log_odds_map'][xis[:-1], yis[:-1]] -= np.log(4)
        MAP['log_odds_map'][xis[-1], yis[-1]] += np.log(4)
        MAP['cell_trajs_map'][sx, sy] = 1

    return MAP['log_odds_map']


def prediction_step(Xt, vt, dt, wt, noise=False) -> np.ndarray:
    """
    :param Xt: pose and yaw angle in rad
    :param vt: linear velocity
    :param dt: time diff
    :param wt: angular velocity
    :param noise:
    :return:
    """
    theta_t = Xt[2, :]
    N = Xt.shape[1]
    if noise:
        cov_vt, cov_wt = 23.792928582405533, 0.002327690853394245
        cov = np.array([[cov_vt, 0],
                        [0, cov_wt]])
        gaussian = np.random.multivariate_normal([0, 0], cov, N).T
        # print(gaussian)
        eps = np.empty_like(Xt)
        eps[0, :] = gaussian[0, :]
        eps[1, :] = gaussian[0, :]
        eps[2, :] = gaussian[1, :]
    else:
        eps = np.zeros(Xt.shape)
    dx = vt * np.cos(theta_t)
    dy = vt * np.sin(theta_t)
    wt = wt * np.ones((1, N))

    dX = np.vstack((dx, dy, wt))
    return Xt + dt * (dX + eps)


def update_step(MAP: dict, particles: np.ndarray, weights: np.ndarray, s_F_: np.ndarray):
    """
    Update particle weights
    :param MAP:
    :param particles:
    :param weights:
    :param s_F_:
    :return:
    """
    map = np.where(MAP['log_odds_map'] > 0, 1, 0).astype(np.int8)
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map
    # x deviation, y deviation
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])
    N = particles.shape[1]
    correlation = np.zeros(N)
    for i in range(N):
        Xt = particles[:, i]
        x_W = Xt[0]
        y_W = Xt[1]
        theta_W = Xt[2]
        W_T_F = np.array(
            [[np.cos(theta_W), -np.sin(theta_W), 0, x_W],
             [np.sin(theta_W), np.cos(theta_W), 0, y_W],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float64)
        # Lidar data from body to world frame
        s_W_ = W_T_F @ s_F_
        # Lidar data in world frame
        ex_W = s_W_[0, :]
        ey_W = s_W_[1, :]
        Y = np.stack((ex_W, ey_W))
        # Calculate correlation
        c = utils.mapCorrelation(map, x_im, y_im, Y, x_range, y_range)
        correlation[i] = np.max(c)
    # Update particle weight
    phi = softmax(correlation)
    weights = weights * phi / np.sum(weights * phi)
    return particles, weights


@jit(nopython=True)
def softmax(x):
    # softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / (np.sum(e_x) + 1e-9)


@jit(nopython=True)
def calculate_N_eff(weights: np.ndarray, N: int):
    """
    :param weights:
    :param N:
    :return: N_eff add 1e-8 to prevent DividebyZero error
    """
    return 1 / (weights.reshape(1, N) @ weights.reshape(N, 1) + 1e-8)


@jit(nopython=True)
def resampling(particles: np.ndarray, weights: np.ndarray, N: int):
    """
    Stratified (low variance) resampling
    :param particles: old particles
    :param weights: old weights
    :param N: num of particles
    :return: new particles, new weights
    """
    new_particles = np.zeros((3, N))
    new_weights = np.ones(N) / N
    j = 0
    c = weights[0]
    for k in range(N):
        u = np.random.uniform(0, 1 / N)
        beta = u + k / N
        while beta > c:
            j = j + 1
            c = c + weights[j]
        new_particles[:, k] = particles[:, j]
    return new_particles, new_weights


def compute_stereo(path_l, path_r, verbose=False):
    image_l = cv2.imread(path_l, 0)
    image_r = cv2.imread(path_r, 0)

    image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
    image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

    image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
    image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
    disparity = stereo.compute(image_l_gray, image_r_gray)

    if verbose:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.imshow(image_l)
        ax1.set_title('Left Image')
        ax2.imshow(image_r)
        ax2.set_title('Right Image')
        ax3.imshow(disparity, cmap='gray')
        ax3.set_title('Disparity Map')
        plt.show()
    return disparity


@jit(nopython=True)
def cam2body(s_C_, V_T_C, F_T_V):
    """
    Convert from camera frame to body frame
    :param s_C:
    :param V_T_C:
    :param F_T_V:
    :return: s_F_
    """
    s_V_ = V_T_C @ s_C_
    s_F_ = F_T_V @ s_V_
    return s_F_

def texture_mapping():
    pass


def main():

    pass


if __name__ == '__main__':
    ###################################################################################
    # Running Config
    VERBOSE = False
    DEAD_RECON = False
    SHOW_CONFIG = False
    if SHOW_CONFIG:
        show_image()
    ###################################################################################
    # Dead Reckoning
    # if DEAD_RECON:
    #     start_dead_Recon = utils.tic("--------DEAD RECKONING--------")
    #     sync_fog_encoder_fname = "data/sync_fog_encoder_left.csv"
    #     traj = dead_reckoning(sync_fog_encoder_fname, expert=False, verbose=VERBOSE)
    #     utils.toc(start_dead_Recon, "Finish Dead_Reckoning")

    ###################################################################################
    start_init = utils.tic("--------INIT SENSOR PARAM--------")
    # Init lidar_param, FOG_param, encoder_param
    lidar_param = get_lidar_param(verbose=VERBOSE)
    FOG_param = get_FOG_param(verbose=VERBOSE)
    encoder_param = get_encoder_param(verbose=VERBOSE)
    left_cam_param = get_left_cam_param()
    right_cam_param = get_right_cam_param()
    print("lidar_param\nFOG_param\nencoder_param")
    # if VERBOSE:
    #     # {V}T{L}
    #     print("lidar2vehicle_param[T]: {}\n{}".format(lidar_param["V_T_L"].shape, lidar_param["V_T_L"]))
    #     # {V}T{F}
    #     print("vehicle2FOG_param[T]: {}\n{}".format(FOG_param["F_T_V"].shape, FOG_param["F_T_V"]))
    #
    # ###################################################################################
    # start_map = utils.tic("--------INIT MAP--------")
    # '''Mapping'''
    # # Assume the map prior is uniform, occupied and free space are equally likely
    # MAP = init_map()
    # # if not DEAD_RECON:
    # #     with open('dead_reckon_traj.npy', 'rb') as f:
    # #         traj = np.load(f)
    # #     plt.figure()
    # #     plt.scatter(traj[0, :], traj[1, :])
    # #     plt.title("Dead Reckoning (No Noise)")
    # #     plt.show()
    # utils.toc(start_map, "Finish Loading Dead_Reckoning Trajectory")
    #
    # ###################################################################################
    # start_lidar = utils.tic("--------LOAD & TRANSFORM DATA--------")
    # '''Stereo Image'''
    # left_image_path = "./data/stereo_images/stereo_left/"
    # right_image_path = "./data/stereo_images/stereo_right/"
    # left_img_fname = sorted([f for f in os.listdir(left_image_path)])
    # left_img_timestamp = sorted([os.path.splitext(f)[0] for f in os.listdir(left_image_path)])
    # right_img_fname = [f for f in os.listdir(right_image_path)]
    # right_img_timestamp = [os.path.splitext(f)[0] for f in os.listdir(right_image_path)]
    # print(f"Num of image form left CAM: {len(left_img_timestamp)}")
    # path_l, path_r = left_image_path + left_img_fname[0], right_image_path + right_img_fname[0]
    #
    # '''Lidar, FOG, Encoder Data'''
    # sync_merge_all = pd.read_csv("data/sync_merge_all_left.csv")
    # print(f"sync_merge_all: {sync_merge_all.shape}")
    #
    # num_state = sync_merge_all.shape[0]
    # lidar_data = sync_merge_all.drop(['timestamp', 'delta_yaw', 'dt', 'wt', 'linear_velocity(m/s)'], axis=1).values
    # vt = sync_merge_all['linear_velocity(m/s)'].values
    # delta_yaw = sync_merge_all['delta_yaw'].values
    # tau = sync_merge_all['dt'].values
    # wt = sync_merge_all['wt'].values
    #
    # cov_vt = np.cov(vt)
    # cov_wt = np.cov(wt)
    # print(f"covariance:({cov_vt}, {cov_wt})")
    # utils.toc(start_map, "Finish loading Data")
    #
    # ###################################################################################
    # '''Init Var'''
    # # At t=0 assume robots locate at (0,0) and orientation 0 -> Xt = np.zeros((3, 1))
    # # N = 1
    # # N = 3
    # # N = 25
    # # N = 50
    # N = 100
    # N_threshold = 5  # 0.2 * N
    # particles = np.zeros((3, N))
    # weights = np.ones(N) / N
    # trajectory = []
    # cell_trajs = []
    # ###################################################################################
    # '''SLAM'''
    # for i in tqdm(range(num_state)):  #
    # # for i in tqdm(range(10000)):
    #     # If Lidar data is not NaN, update map
    #     ranges = lidar_data[i, :]
    #     if not np.isnan(np.sum(ranges)):
    #         # Convert LiDAR scan from polar to cartesian coord attached z axis with zeros
    #         s_L = polar2xyz(ranges, lidar_param, verbose=False)
    #         s_F_ = lidar2body(s_L_=reg2homo(s_L), V_T_L=lidar_param["V_T_L"], F_T_V=FOG_param["F_T_V"])
    #         ''' Update_Step '''
    #         particles, weights = update_step(MAP, particles, weights, s_F_)
    #         N_eff = calculate_N_eff(weights, N)
    #         if N_eff < N_threshold:
    #             particles, weight = resampling(particles, weights, N)
    #         # Find particle with largest weight
    #         max_weight = np.argmax(weights)
    #         Xt = particles[:, max_weight]
    #         # Record trajectory
    #         trajectory.append(Xt)
    #         # update log_odds map
    #         MAP['log_odds_map'] = update_map(MAP, Xt, s_F_, verbose=False)
    #         # Resample the particles
    #
    #     if i % 100000 == 0 and i != 0:
    #         show_map(np.where(MAP['log_odds_map'] > 0, 1, 0).astype(np.int8))
    #     '''Prediction Step'''
    #     particles = prediction_step(particles, vt[i], tau[i], wt[i], noise=True)
    # utils.toc(start_map, "Finish MAP Creation & Update MAP log-odds")
    # ##################################################################################
    # # Model the map cells mi as independent Bernoulli random variables
    # MAP['map'] = np.where(MAP['log_odds_map'] > 0, 1, 0).astype(np.int8)
    # show_map(MAP['map'], title="Occupancy grid map")
    # show_map(MAP["cell_trajs_map"], title="cell_trajs")
    # show_map(MAP['map'] + MAP["cell_trajs_map"], title="map&trajs")
    # # Save result
    # with open('map_test_100_tresh_5.pkl', 'wb') as f:
    #     pickle.dump(MAP, f)
    # with open('map_test_100_tresh_5.pkl', 'rb') as f:
    #     MAP= pickle.load(f)
    #     show_map(MAP['map'].astype(np.int8), title="Occupancy grid map")
    #     show_map(MAP["cell_trajs_map"], title="cell_trajs")
    #     show_map((MAP['map'] + MAP["cell_trajs_map"]).astype(np.int8), title="map&trajs")
    # show_map(MAP['map'])

    ##################################################################################
    # trajectory = np.vstack(trajectory)
    # trajectory = trajectory.T
    # with open('test_traj.npy', 'wb') as f:
    #     np.save(f, trajectory)
    # plt.figure()
    # plt.scatter(trajectory[0, :], trajectory[1, :])
    # plt.title("trajectory (with Noise)")
    # plt.show()
    # cell_trajs = np.hstack(cell_trajs)
    # with open('cell_traj.npy', 'wb') as f:
    #     np.save(f, cell_trajs)
    # with open('cell_traj.npy', 'rb') as f:
    #     cell_trajs = np.load(f)
    #################################################################################
    start_cam = utils.tic("--------Texture Mapping--------")
    '''
    1. Find the depth of each pixel in the left camera.
    2. Find the 3D Cartesian coordinates of each pixel in the left camera frame.
    3. Transform the pixels from the camera frame to the world frame using
        the best (highest weight) particle at each time step.
        Find the map cells that these points fall in.
        You might want to consider only the pixels whose z coordinate in the world frame falls
        within a small plus/minus range of the flat plane of the map,
        i.e., do not store colors of things that are way above or way below the plane that you are mapping.
    4.Associate the RGB values with the corresponding map cells.
     It is fine to over-write previous RGB values and it is not necessary to do any kind of smoothing or interpolation.
    '''
    PR = right_cam_param["projection_matrix"]
    b = left_cam_param["baseline"]
    fsu = PR[0, 0]
    fsv = PR[1, 1]
    cu = PR[0, 2]
    cv = PR[1, 2]
    # d = compute_stereo(path_l, path_r, verbose=False)
    # print("d:", d.shape)
    # z = fsu*b/(d + 1e-8)
    # # (560, 1280)
    # a= np.array(range(1280)).reshape(1, -1)
    # b = np.array(range(560)).reshape(1, -1)
    # # print("uL vector", a.shape)
    # # print("vL vector", b.shape)
    #
    # uL = np.tile(a, (560, 1))
    # vL = np.tile(b, (1280, 1)).T
    # # print("uL", uL.shape)
    # # print("vL", vL.shape)
    #
    # x = (uL-cu)/fsu * z
    # y = (vL-cv)/fsv * z
    # # print("x", x.shape)
    # # print("y", y.shape)
    # # print("z", z.shape)
    #
    # x_vec = x.flatten()
    # y_vec = y.flatten()
    # z_vec = z.flatten()
    # s_C = np.vstack((x_vec, y_vec, z_vec))
    # # print(s_C.shape)
    #
    # # Transform form camera frame to fog frame s_C-> s_V ->s_F
    # V_T_C = left_cam_param["V_T_C"]
    # F_T_V = FOG_param["F_T_V"]
    #
    # s_F = cam2body(reg2homo(s_C), V_T_C, F_T_V)
    # utils.toc(start_cam, "Texture Mapping")