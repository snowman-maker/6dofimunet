import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, cos, sin, floor
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2)  
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, LSTM, concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
# from keras_flops import get_flops
import pickle
import csv
import random
import itertools
import math
import time
import sys
import yaml
import argparse
from tqdm import tqdm
import colorlog
import logging
from datetime import datetime
from conf import configer
from os import path as osp
from scipy.fft import fft

color = colorlog.Log()
logging.basicConfig(
	stream=sys.stdout,
	format=color.green("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s --> %(message)s"),
	level=logging.INFO,
)

def getConfig(cfg_file):
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def GetTimeNow():
	now = datetime.now()
	# 提取年、月、日、小时和分钟
	year = now.year
	month = now.month
	day = now.day
	hour = now.hour
	minute = now.minute
	return year, month, day, hour, minute

def create_output_dir(out_dir):
	try:
		if out_dir is not None:
			if not os.path.isdir(out_dir):
				os.makedirs(out_dir)
			if not os.path.isdir(os.path.join(out_dir, "checkpoints")):
				os.makedirs(os.path.join(out_dir, "checkpoints"))
			if not os.path.isdir(os.path.join(out_dir, "logs")):
				os.makedirs(os.path.join(out_dir, "logs"))
			logging.info(f"Training output writes to {out_dir}")
			return out_dir
		else:
			raise ValueError("out_dir must be specified.")
	except ValueError as e:
		logging.error(e)
		return

def GetDataPath(path):
	names = os.listdir(path + "/")
	folders = []
	for name in names:
		data_path = os.path.join(os.path.abspath(path), name)
		if os.path.isdir(data_path):
			folders.append(data_path)
	folders.sort()
	return folders

def dataHandle(paths, cfg):
    x0_list = []
    y0_list = []
    z0_list = []
    q0_list = []
    size_of_each = []
    window_size = int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])
    stride = int(cfg['model_param']['stride'])
    X = np.empty([0, window_size, 6]) # acc, gyro
    Y_pos = np.empty([0, window_size, 7]) # x, y, z, w, x, y, z

    x_vel = np.empty([0])
    y_vel = np.empty([0])
    z_vel = np.empty([0])

    Physics_Vec = np.empty([0])

    for path in tqdm(paths):
        if path[-1] == '/':
            path = path[:-1]

        file = osp.join(path, 'data.h5')
        if osp.exists(file):
            imu_data = pd.read_hdf(file, 'data')
        else:
            logging.info(f"file {file} is not exist. ")
            return
        # unit of ground truth: meters
        cur_GT = np.array(imu_data[['gt_p_x', 'gt_p_y', 'gt_p_z', 'gt_q_w', 'gt_q_x', 'gt_q_y', 'gt_q_z']].values)
        #Take care of missing data
        for i in range(cur_GT.shape[1]):
            mask = np.isnan(cur_GT[:,i])
            cur_GT[mask,i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cur_GT[~mask,i])
        #unit of IMU and compass data: acc, gyro
        tmp_train = np.array(imu_data[['acce_x','acce_y','acce_z','gyro_x','gyro_y','gyro_z']].values)
        tmp_bias = np.array(imu_data[['acce_bias_x','acce_bias_y','acce_bias_z','gyro_bias_x','gyro_bias_y','gyro_bias_z']].values)
        cur_train = tmp_train - tmp_bias
        #take care of missing data
        ind = np.where(~np.isnan(cur_train))[0]
        first, last = ind[0], ind[-1]
        cur_train[:first] = cur_train[first]
        cur_train[last + 1:] = cur_train[last]
        windows = SlidingWindow(size=window_size, stride=stride)
        #Window IMU Readings
        cur_train_3D = windows.fit_transform(cur_train[:,0])
        for i in range(1,cur_train.shape[1]):
            X_windows = windows.fit_transform(cur_train[:,i])
            cur_train_3D = np.dstack((cur_train_3D,X_windows))
        #Window Ground Truth
        cur_GT_3D = windows.fit_transform(cur_GT[:,0])
        for i in range(1,cur_GT.shape[1]):
            X_windows = windows.fit_transform(cur_GT[:,i])
            cur_GT_3D = np.dstack((cur_GT_3D,X_windows))

        # Extract Physics Channel
        loc_mat = np.empty((cur_train_3D.shape[0]))
        for i in range(cur_train_3D.shape[0]):
            acc_x =  cur_train_3D[i,:,0]
            acc_y =  cur_train_3D[i,:,1]
            acc_z =  cur_train_3D[i,:,2]
            VecSum = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            VecSum = VecSum - np.mean(VecSum)
            FFT_VS = fft(VecSum)
            P2 = np.abs(FFT_VS/acc_x.shape[0])
            P1 = P2[0:math.ceil(acc_x.shape[0]/2)]
            P1[1:-1-2] = 2*P1[1:-1-2]
            loc_mat[i] = np.mean(P1)
        
        # Extract Ground Truth Velocity
        vx = np.zeros((cur_GT_3D.shape[0]))
        vy = np.zeros((cur_GT_3D.shape[0]))
        vz = np.zeros((cur_GT_3D.shape[0]))
        for i in range(cur_GT_3D.shape[0]): 
            Xdisp = (cur_GT_3D[i,-1,0]-cur_GT_3D[i,0,0])
            vx[i] = Xdisp
            Ydisp = (cur_GT_3D[i,-1,1]-cur_GT_3D[i,0,1])
            vy[i] = Ydisp
            Zdisp = (cur_GT_3D[i,-1,2]-cur_GT_3D[i,0,2])
            vz[i] = Zdisp
        
        #Stack readings
        X = np.vstack((X, cur_train_3D))
        Physics_Vec = np.concatenate((Physics_Vec, loc_mat))
        Y_pos = np.vstack((Y_pos, cur_GT_3D))
        x0_list.append(cur_GT[0,0])
        y0_list.append(cur_GT[0,1])
        z0_list.append(cur_GT[0,2])
        q0_list.append(cur_GT[0,3:])
        size_of_each.append(cur_GT_3D.shape[0])
        x_vel = np.concatenate((x_vel, vx))
        y_vel = np.concatenate((y_vel, vy))
        z_vel = np.concatenate((z_vel, vz))
    return X, Y_pos, Physics_Vec, x_vel, y_vel, z_vel, x0_list, y0_list, z0_list, q0_list, size_of_each

def LoadData(cfg, flag=1):
    if(flag == 1):
        train_data_path = GetDataPath(cfg['data']['train_dir'])
        logging.info(f'Processing train data, total nums: {len(train_data_path)}')
        imu_data, gt_data, Physics_Vec, x_vel, y_vel, z_vel, x0_list, y0_list, z0_list, q0_list, size_of_each = dataHandle(train_data_path, cfg)
        return imu_data, gt_data, Physics_Vec, x_vel, y_vel, z_vel, x0_list, y0_list, z0_list, q0_list, size_of_each
    elif(flag == 0):
        test_data_path = GetDataPath(cfg['data']['test_dir'])
        logging.info(f'Processing test data, total nums: {len(test_data_path)}')
        imu_data, gt_data, Physics_Vec, x_vel, y_vel, z_vel, x0_list, y0_list, z0_list, q0_list, size_of_each = dataHandle(test_data_path, cfg)
        return imu_data, gt_data, Physics_Vec, x_vel, y_vel, z_vel, x0_list, y0_list, z0_list, q0_list, size_of_each
    else:
        print("You must specify train or test")
        return
    
    
def loadDataset(cfg):
    window_size = int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])

    train_data, gt_train, physics_vec_train, x_vel_train, y_vel_train, z_vel_train, x0_list_train, y0_list_train, z0_list_train, q0_list_train, size_of_each_train = LoadData(cfg, flag=1)

    P = np.repeat(physics_vec_train, window_size).reshape((physics_vec_train.shape[0],window_size,1))
    train_data = np.concatenate((train_data,P), axis=2)

    test_data, gt_test, physics_vec_test, x_vel_test, y_vel_test, z_vel_test, x0_list_test, y0_list_test, z0_list_test, q0_list_test, size_of_each_test= LoadData(cfg, flag=0)
    P_test = np.repeat(physics_vec_test,window_size).reshape((physics_vec_test.shape[0],window_size,1))
    test_data = np.concatenate((test_data,P_test),axis=2)


def main(cfg):
    loadDataset(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgfile', help='Please input config file', default='./conf/default.yaml')
    args = parser.parse_args()
    logging.info(f'use yaml file: {args.cfgfile}')
    
    gpus = tf.config.list_physical_devices('GPU')
    logging.info(f"Model is loaded to {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    cfg = configer.LoadConfig(args.cfgfile)
    with open(cfg['model']['model_yaml'], 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    configer.UpdateRecursive(cfg, cfg_special)

    if cfg['seeds']['use_seeds']:
        set_seeds(cfg['seeds']['id'])
        logging.info(f"use seeds: {cfg['seeds']['id']}")

    if cfg['schemes']['train']:
        year, month, day, hour, minute = GetTimeNow()
        out_dir = create_output_dir(cfg['train']['out_dir'] + f'{year}{month}{day}{hour}{minute}')
        # copy yaml
        cmd = "cp " + args.cfgfile + " " + f'{out_dir}' + "/default.yaml"
        os.system(cmd)
        cmd = "cp " + cfg['model']['model_yaml'] + " " + f'{out_dir}' + "/model.yaml"
        os.system(cmd)

        if cfg['data']['train_dir'] is None:
            raise ValueError("train_dir must be specified.")
    # print(cfg)
    main(cfg)

