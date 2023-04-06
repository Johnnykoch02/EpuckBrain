import numpy as np
import os

def save_cell_prediction_npz(obs, path):
    # Initialize the output data arrays
    theta_data = []
    lidar_data = []
    current_cell_data = []
    cameraFront_data = []
    cameraRear_data = []
    cameraLeft_data = []
    cameraRight_data = []
    current_cell_data = []

    # Loop through the observations in the list
    for obs_dict in obs:
        try:
            obs_dict['theta']
        except:
            continue
        theta_data.append(obs_dict['theta'])
        lidar_data.append(obs_dict['lidar'])
        current_cell_data.append(obs_dict['current_cell'])
        cameraFront_data.append(obs_dict['cameraFront'])
        cameraRear_data.append(obs_dict['cameraRear'])
        cameraLeft_data.append(obs_dict['cameraLeft'])
        cameraRight_data.append(obs_dict['cameraRight'])
    
    # Save the data to an NPZ file
    np.savez(path, theta=theta_data, lidar=lidar_data, 
                        current_cell=current_cell_data, cameraFront=cameraFront_data,
                        cameraRear=cameraRear_data, cameraLeft=cameraLeft_data,
                        cameraRight=cameraRight_data)
    
def load_cell_prediction_npz(save_dir, noise_level=0.01):
    # Initialize the output data arrays
    theta_data = []
    lidar_data = []
    current_cell_data = []
    cameraFront_data = []
    cameraRear_data = []
    cameraLeft_data = []
    cameraRight_data = []

    # Loop through the files in the save_dir
    for file_path in os.listdir(save_dir):
        npz_file = np.load(os.path.join(save_dir, file_path))
        for data in npz_file['theta']:
            theta_data.append(data)
        for data in npz_file['lidar']:
            lidar_data.append(data)
        for data in npz_file['cameraFront']:
            cameraFront_data.append(data)
        for data in npz_file['cameraRear']:
            cameraRear_data.append(data)
        for data in npz_file['cameraLeft']:
            cameraLeft_data.append(data)
        for data in npz_file['cameraRight']:
            cameraRight_data.append(data)
        for data_idx in range(0, len(npz_file['current_cell']), 2): #FIX LATER
            current_cell_data.append(npz_file['current_cell'][data_idx])    
    # Init the data
    theta_data_norm = np.nan_to_num(np.array(theta_data))
    lidar_data_norm = np.nan_to_num(np.array(lidar_data) )
    cameraFront_data_norm = np.array(cameraFront_data)
    cameraRear_data_norm = np.array(cameraRear_data)
    cameraLeft_data_norm = np.array(cameraLeft_data)
    cameraRight_data_norm = np.array(cameraRight_data)
    current_cell_data = np.array(current_cell_data) - 1
    
    # # Reshape the data into a rectangular matrix.
    # theta_data_norm = np.reshape(theta_data_norm, (-1, theta_data_norm[0].shape[0]))
    # lidar_data_norm = np.reshape(lidar_data_norm, (-1, lidar_data_norm[0].shape[0]))
    # current_cell_data = np.reshape(current_cell_data, (-1, current_cell_data[0].shape[0]))
    # cameraFront_data_norm = np.reshape(cameraFront_data_norm, (-1, cameraFront_data_norm[0].shape[1], cameraFront_data_norm[0].shape[2], cameraFront_data_norm[0].shape[2]))
    # cameraRear_data_norm = np.reshape(cameraRear_data_norm, (-1, cameraRear_data_norm[0].shape[1], cameraRear_data_norm[0].shape[2], cameraRear_data_norm[0].shape[2]))
    # cameraLeft_data_norm = np.reshape(cameraLeft_data_norm, (-1, cameraLeft_data_norm[0].shape[1], cameraLeft_data_norm[0].shape[2], cameraLeft_data_norm[0].shape[2]))
    # cameraRight_data_norm = np.reshape(cameraRight_data_norm, (-1, cameraRight_data_norm[0].shape[1], cameraRight_data_norm[0].shape[2], cameraRight_data_norm[0].shape[2]))
    
    # Normalize the data
    theta_data_norm = np.array(theta_data) / np.max(theta_data)
    lidar_data_norm = np.array(lidar_data) / 40
    cameraFront_data_norm = np.array(cameraFront_data) / np.max(cameraFront_data)
    cameraRear_data_norm = np.array(cameraRear_data) / np.max(cameraRear_data)
    cameraLeft_data_norm = np.array(cameraLeft_data) / np.max(cameraLeft_data)
    cameraRight_data_norm = np.array(cameraRight_data) / np.max(cameraRight_data)
    
    lidar_data_norm = np.clip(lidar_data_norm, 0, 1)
    
    # Add noise to the data
    theta_data_norm += noise_level * np.random.normal(size=theta_data_norm.shape)
    lidar_data_norm += noise_level * np.random.normal(size=lidar_data_norm.shape)
    cameraFront_data_norm += noise_level * np.random.normal(size=cameraFront_data_norm.shape)
    cameraRear_data_norm += noise_level * np.random.normal(size=cameraRear_data_norm.shape)
    cameraLeft_data_norm += noise_level * np.random.normal(size=cameraLeft_data_norm.shape)
    cameraRight_data_norm += noise_level * np.random.normal(size=cameraRight_data_norm.shape)
    
    
    
    # Return the normalized arrays with the data
    return {'theta':theta_data_norm, 'lidar':lidar_data_norm, 'cameraFront': cameraFront_data_norm, 'cameraRear':cameraRear_data_norm, 'cameraLeft':cameraLeft_data_norm, 'cameraRight': cameraRight_data_norm}, current_cell_data