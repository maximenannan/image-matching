from typing import Tuple

import numpy as np
import scipy
import matplotlib.pyplot as plt

def generate_points(N: int) -> np.ndarray:
    """It randomly generates N points in [-1, 1] x [-1, 1]"""
    return 2*(np.random.random((N, 2)) - 0.5)


def transform_points(
        points: np.ndarray, theta: float = 0, scale: float = 1, sigma: float =1e-5
) -> np.ndarray:
    """
    Given a set of points np.ndarray of shape (N,2).
    This function adds some gaussian noise with a variance of sigma, apply a rotation of theta (in radians)
    and a scaling of a factor scale.
    """
    N = points.shape[0]
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] )
    noise = np.random.normal(loc=0, scale=sigma, size=(N, 2))
    return np.dot(points, rotation_matrix.T)*scale + noise


def compute_features(triangles: np.ndarray) -> np.ndarray:
    """
    Given a numpy ndarray of triangles this returns the features computed from these triangles/
    :param triangles:
    :return:
    """
    # input shape (-1, 3, 2)
    vect = np.concatenate([
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 1],
        triangles[:, 0] - triangles[:, 2]
    ], axis=1).reshape((-1, 3, 2))
    norm = np.linalg.norm(vect, axis=2).reshape((-1, 3, 1))
    normalized_vect = vect/(norm + 1e-12)
    return np.concatenate([
        np.linalg.det(normalized_vect[:, [0, 2], :]).reshape((-1, 1)),
        np.linalg.det(normalized_vect[:, [1, 0], :]).reshape((-1, 1)),
        np.linalg.det(normalized_vect[:, [2, 1], :]).reshape((-1, 1)),
    ], axis=1)


def generate_dataset(
        N: int, theta: float, scale: float, sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an integer N, it generates N points in [-1, 1] x [-1, 1] and applies a rotation,
    a scaling and add some noise.
    Finally it returns N generated points and their transformation.
    :param N: Number of points in the dataset
    :param theta: Angle for the rotation (in radians)
    :param scale: Scaling factor
    :param sigma: Noise variance
    :return:
    """
    data1 = generate_points(N)
    data2 = transform_points(data1, theta, scale, sigma)
    return data1, data2


def get_closest_features(
        features1: np.ndarray, features2: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two set of features, for each feature in features1 it retrieves the n nearest features in
    features2.
    :param features1: numpy ndarray of features
    :param features2: numpy ndarray of features
    :param n: number of nearest features to extract
    :return: A tuple of numpy array of size (N1,n) which represents the distance between features
    of features1 and the n nearest in features2 and a numpy array of the indexes of the nearest
    features in features2.
    """
    tree = scipy.spatial.KDTree(features2.reshape((-1, 3)))
    nearest_distances, nearest_neighbors = tree.query(features1.reshape((-1, 3)), n)
    return nearest_distances, nearest_neighbors


def plot_solution(
        points1: np.ndarray, points2:np.ndarray, mapping:np.ndarray
):
    fig = plt.figure(figsize=(8,8))
    plt.scatter(points1[:,0], points1[:,1], marker='+')
    plt.scatter(points2[mapping][:,0], points2[mapping][:, 1], marker='+', color='r')
    for (x1,y1), (x2,y2) in zip(points1, points2[mapping]):
        plt.plot([x1,x2], [y1, y2], color='b')
    for (x1,y1), (x2,y2) in zip(points1, points2):
        plt.plot([x1,x2], [y1, y2], color='r')


def power_iteration(
        triangles1_indexes: np.ndarray, nearest_neighbors_indexes: np.ndarray,
        potentials: np.ndarray,  N1:int, N2:int, eps = 1e-5
) -> np.ndarray:
    maxiter = 100
    Vn_1 = np.ones((N1, N2))
    for epoch in range(maxiter):
        print(f'Epoch {epoch+1}')
        Vn = np.ones((N1, N2))
        for i1 in range(N1):
            for t1, (_, j1, k1) in enumerate(triangles1_indexes[i1]):
                for t2, (i2, j2, k2) in enumerate(nearest_neighbors_indexes[i1][t1]):
                    Vn[i1, i2] += potentials[i1, t1, t2] * Vn_1[j1, j2] * Vn_1[k1, k2]
                    Vn[j1, j2] += potentials[i1, t1, t2] * Vn_1[i1, i2] * Vn_1[k1, k2]
                    Vn[k1, k2] += potentials[i1, t1, t2] * Vn_1[j1, j2] * Vn_1[i1, i2]
        Vn = Vn / np.linalg.norm(Vn)
        print(f'||Vn - Vn_1|| = {np.linalg.norm(Vn - Vn_1)}')
        if np.linalg.norm(Vn - Vn_1) < eps:
            break
        Vn_1 = Vn
    print(f'Stop after {epoch} iterations')
    mapping = np.argmax(Vn, axis=1)
    return mapping
