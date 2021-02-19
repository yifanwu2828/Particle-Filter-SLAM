from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pr2_utils as utils


def split2two(input_data: np.ndarray):
    """
    split data into timestamp, data
    """
    return input_data[:, 0], input_data[:, 1:]


def show_image() -> None:
    """
    Show Vehicle dimensions and coordinates
    """
    vehicle_img = plt.imread("data/vehicle_cofig.png")
    plt.imshow(vehicle_img)
    plt.show()


def show_lidar(angles: np.ndarray, ranges: np.ndarray) -> None:
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
    ax.set_title("Lidar scan data", va='bottom')
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


def polar2xyz(lidar_data: np.ndarray, lidar_param: dict, index=0, verbose=False) -> np.ndarray:
    """
    Convert from polar coordinates to cartesian coordinate with z axis fill with zeros
    Remove scan points that are too close or too far,
    Only consider points between [min_range=2, max_range=80]
    * Measurements between 2m-75m are recommended to be included as valid data.
    :param: lidar_data
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
    assert isinstance(lidar_data, np.ndarray)
    assert isinstance(index, int)
    assert isinstance(lidar_param, dict)
    max_range = lidar_param["max_range"]
    min_range = lidar_param["min_range"]
    # angles = np.deg2rad(np.linspace(-5, 185, 286))
    angles = lidar_param["angles"]
    ranges = lidar_data[0, :]
    if verbose:
        show_lidar(angles, ranges)
    # Filter out noisy data (r<2 and r>80)
    indValid = np.logical_and((ranges <= max_range), (ranges >= min_range))
    ranges = ranges[indValid]
    angles = angles[indValid]
    if verbose:
        show_lidar(angles, ranges)
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


def reg2homo(X: np.ndarray) -> np.ndarray:
    """
    Convert Matrix to homogenous coordinate
    :param X: matrix/vector
    :type :numpy array
    return X_ -> [[X]
                  [1]]
    """
    assert isinstance(X, np.ndarray)
    ones = np.ones((1, X.shape[1]), dtype=np.float64)
    # print(ones)
    X_ = np.concatenate((X, ones), axis=0)
    return X_


def transform(s_L, b_R_l, pos) -> np.ndarray:
    """
    Covert from Lidar frame to vehicle frame
    :param s_L: point cloud from laser scanner in regular coordinates will transform to homogenous
    :type :numpy.array
    :param b_R_l: rotation matrix from lidar frame to body frame
    :type : numpy.array
    :param pos: position in vehicle frame
    :type : numpy.array
    :return: s_V [x,y,z].T coordinate in vehicle frame
    """
    assert isinstance(s_L, np.ndarray)
    assert isinstance(b_R_l, np.ndarray)
    assert isinstance(pos, np.ndarray)
    assert pos.ndim == 1

    T_l = get_T(b_R_l, pos)
    # print(f"B_T_L: {T_l}")
    ''' s_V = T{L} @ s_L in homogenous coord'''
    s_V = T_l @ reg2homo(s_L)  # Transform to homogenous coord
    # remove dummy ones [x,y,z].T->[x,y,z,1].T: 4x286 -> 3x286
    s_V = np.delete(s_V, 3, axis=0)
    # print(s_V.shape) # 3x286
    return s_V


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


def dead_reckoning(path, verbose=False):
    """
    Perform dead_reckoning
    :param path: path to sync_fog_encoder_df
    :type: str
    :param verbose: bool
    return (x_min, x_max), (y_min, y_max)
    """
    sync_fog_encoder_df = pd.read_csv(path)
    total_duration = (max(sync_fog_encoder_df['timestamp']) - min(sync_fog_encoder_df['timestamp'])) * 1e-3
    print(f"total_duration:{total_duration} secs")
    assert abs(sum(sync_fog_encoder_df['dt']) - total_duration) < 1
    num_state = sync_fog_encoder_df.shape[0]
    vt = np.array(sync_fog_encoder_df['linear_velocity(m/s)'], dtype=np.float64)
    delta_yaw= np.array(sync_fog_encoder_df['delta_yaw'], dtype=np.float64)
    tao = np.array(sync_fog_encoder_df['dt'], dtype=np.float64)
    if verbose:
        print(f"sync_fog_encoder_df: {sync_fog_encoder_df.shape}")
        print(sync_fog_encoder_df.head(5))
    # Assume initial state (x:0,y:0,theta:0)
    X0 = np.zeros((3, 1))
    lst = []
    for i in range(num_state):
        lst.append(X0)
        X0 = differential_drive_model(X0, vt[i], delta_yaw[i], tao[i])
    trajectory = np.hstack(lst)
    plt.figure()
    plt.scatter(trajectory[0, :], trajectory[1, :])
    plt.title("Dead Reckoning (No Noise)")
    plt.show()
    (x_min, x_max) = np.min(trajectory[0, :]), np.max(trajectory[0, :])
    (y_min, y_max) = np.min(trajectory[1, :]), np.max(trajectory[1, :])
    return (x_min, x_max), (y_min, y_max)


def main():
    pass


if __name__ == '__main__':
    # Running Config
    np.seterr(all='raise')
    pd.set_option("precision", 10)
    VERBOSE = True
    if VERBOSE:
        pd.pandas.set_option('display.max_columns', None)

    # Dead Reckoning
    start_load = utils.tic()
    sync_fog_encoder_fname = "data/sync_fog_encoder_left.csv"
    if VERBOSE:
        show_image()
        start_dead_Recon = utils.tic("--------DEAD-RECKONING--------")
        dead_reckoning(sync_fog_encoder_fname, verbose=VERBOSE)
        utils.toc(start_dead_Recon, "Finish dead_Reckoning")
    utils.toc(start_load, "Finish program")

    ###################################################################################
    # Init lidar_param, FOG_param, encoder_param
    lidar_param = get_lidar_param(verbose=False)
    FOG_param = get_FOG_param(verbose=False)
    encoder_param = get_encoder_param(verbose=False)

    R = lidar_param["V_Rot_L"]
    p = lidar_param["V_pos_L"]
    print(f"lidar2vehicle_param[R]: {R.shape}\n{R}")
    print(f"lidar2vehicle_param[p]: {p.shape}\n{p}\n")
    utils.toc(start_load, "Initiate params")

    ###################################################################################
    '''
    Use the first laser scan to initialize and display the map to make sure your transforms are correct:
    1. convert the scan to cartesian coordinates
    2. transform the scan from the lidar frame to the body frame and then to the world frame
        At t=0, you can assume that the body frame and the world frame are perfectly aligned.
        For t>0 you need to localize the robot in order to find the transformation between body and world.
    3. convert the scan to cells (via bresenham2D or cv2.drawContours) and update the map log-odds
    '''
    # Convert LiDAR scan from polar to cartesian coord attached z axis with zeros -- step1
    _, lidar_data = utils.read_data_from_csv('data/sensor_data/lidar.csv')
    s_L0 = polar2xyz(lidar_data, lidar_param, index=0, verbose=False)
    print(f"s_L[0]: {s_L0.shape}")

    # Transform from lidar frame to vehicle frame s_B in [x,y,z].T
    s_V0 = transform(s_L0, R, p)  # s_L -> s_V
    print(f"s_V[0]: {s_V0.shape}")

    # Define FOG frame to be the body frame, coincide with vehicle frame
    p_B = FOG_param["V_pos_F"]
    s_B0 = s_V0 - p_B.reshape(3, 1)  # s_V -> s_F
    print(f"s_B[0]: {s_V0.shape}")

    # Body frame and the world frame are perfectly aligned t=0 -> s_W = I@s_B + 0 -- step2
    s_W0 = s_B0
    print(f"s_W[0] with z-axis: {s_W0.shape}")
    # Remove z-axis
    s_W0 = np.delete(s_W0, 2, axis=0)
    print(f"s_W[0] without z-axis: {s_W0.shape}")
    utils.toc(start_load, "Transform from laser to body s_L -> s_B -> s_W at t = 0")

    ######################################################################################
    # At t=0 assume robots locate at (0,0) and orientation 0
    # TODO: step3: convert the scan to cells (via bresenham2D or cv2.drawContours) and update the map log-odds
    # Assign each point to a specific cell in the map and then do bresenham2D
    # convert nx(x,y) to row and columns
    x_min, x_max = 0.0, 1238.0
    y_min, y_max = -1012.0, 0.0
    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"y_min: {y_min}, x_max: {y_max}")

    # init MAP
    MAP = dict()
    MAP['res'] = 0.5  # meters
    MAP['xmin'] = x_min - lidar_param["max_range"]  # meters
    MAP['ymin'] = y_min - lidar_param["max_range"]
    MAP['xmax'] = x_max + lidar_param["max_range"]
    MAP['ymax'] = y_max + lidar_param["max_range"]
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8

    xs0 = s_W0[0, :]
    ys0 = s_W0[1, :]
    show_laserXY(xs0, ys0)

    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # plt.matshow(a)
    # plt.show()
