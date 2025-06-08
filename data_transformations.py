'''data_transformations.py
Susie Mueller
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    normalizedData = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
    return normalizedData


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    means = np.mean(data, axis = 0) # calculate mean of each column
    centeredData = data - means
    return centeredData

def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    matrix = np.eye(3) 
    theta = np.deg2rad(degrees)
    if axis == 'x': 
        matrix[1, 1] = np.cos(theta)
        matrix[2, 1] = np.sin(theta)
        matrix[1, 2] = - np.sin(theta)
        matrix[2, 2] = np.cos(theta)
    if axis == 'y': 
        matrix[0, 0] = np.cos(theta)
        matrix[0, 2] = np.sin(theta)
        matrix[2, 0] = - np.sin(theta)
        matrix[2, 2] = np.cos(theta)
    if axis == 'z': 
        matrix[0, 0] = np.cos(theta)
        matrix[1, 0] = np.sin(theta)
        matrix[0, 1] = - np.sin(theta)
        matrix[1, 1] = np.cos(theta)
    return matrix
