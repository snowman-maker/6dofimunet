from scipy.interpolate import interp1d
import pandas
import random
from numpy.random import normal as gen_normal
from os import path as osp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import logging

class Sequence(object):
	"""
	Reading and processing sequence data.
	"""
	def __init__(self, data_path, file_name, imu_freq, window_size, verbose=True):
		super().__init__()
		(
			self.timestamps,
			self.velocity,
			self.acceleration,
			self.gyroscope,
			self.features,
			self.targets,
			self.orientations,
			self.gt_pos,
			self.gt_ori,
			self.sampling_time,
		) = (None, None, None, None, None, None, None, None, None, None)

		# temp variables
		self.tmp_acce = None
		self.tmp_gyro = None
		self.tmp_gt_p = None
		self.tmp_gt_q = None
		self.tmp_ts = None
		self.tmp_velocity = None

		self.imu_freq = imu_freq
		self.interval = window_size
		self.data_valid = False
		self.sum_dur = 0
		self.valid = False
		self.file_name = file_name

		if data_path is not None:
			self.valid = self.Load(data_path, verbose=verbose)

	def LoadTemp(self, data_path):
		if data_path[-1] == '/':
			data_path = data_path[:-1]

		file = osp.join(data_path, self.file_name)
		if osp.exists(file):
			imu_all = pandas.read_hdf(file, 'data')
		else:
			logging.info(f"file {file} is not exist. ")
			return

		tmp_ts = np.array(imu_all[['timestamps']].values)

		if tmp_ts.shape[0] < 1000:
			return False
		self.tmp_ts = np.squeeze(tmp_ts)
		self.sampling_time = np.array(tmp_ts[1:] - tmp_ts[:-1])

		self.tmp_velocity = np.array(imu_all[['v_x', 'v_y', 'v_z']].values)

		self.tmp_gt_q = np.array(imu_all[['gt_q_w', 'gt_q_x', 'gt_q_y', 'gt_q_z']].values)
		self.tmp_gt_p = np.array(imu_all[['gt_p_x', 'gt_p_y', 'gt_p_z']].values)

		tmp_gyro = np.array(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
		tmp_accel = np.array(imu_all[['acce_x', 'acce_y', 'acce_z']].values)

		tmp_gyro_bias = np.array(imu_all[['gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z']].values)
		tmp_acce_bias = np.array(imu_all[['acce_bias_x', 'acce_bias_y', 'acce_bias_z']].values)

		# self.tmp_gyro = tmp_gyro - tmp_gyro_bias[-1, :]
		# self.tmp_acce = tmp_accel - tmp_acce_bias[-1, :]

		self.tmp_gyro = tmp_gyro - tmp_gyro_bias
		self.tmp_acce = tmp_accel - tmp_acce_bias

	def Load(self, data_path, verbose=True):
		"""
		Discard the first 10 frames and the last 20 frames.
		Linearize and interpolate the original data.
		"""
		self.LoadTemp(data_path)
		# 为了消除采样时间对数据的影响,保证了采样时间为 1.0 / self.imu_freq
		start_ts = self.tmp_ts[10]
		end_ts = self.tmp_ts[10] + int((self.tmp_ts[-20] - self.tmp_ts[1]) * self.imu_freq) / self.imu_freq
		ts = np.arange(start_ts, end_ts, 1.0 / self.imu_freq)
		self.data_valid = True
		self.sum_dur = end_ts - start_ts

		if verbose:
			logging.info(f"{data_path}: Sum duration: {self.sum_dur}s")

		q_slerp = Slerp(self.tmp_ts, Rotation.from_quat(self.tmp_gt_q[:, [1, 2, 3, 0]]))  # x, y, z, w
		r = q_slerp(ts)
		p_interp = interp1d(self.tmp_ts, self.tmp_gt_p, axis=0)(ts)

		gyro = interp1d(self.tmp_ts, self.tmp_gyro, axis=0)(ts)
		acce = interp1d(self.tmp_ts, self.tmp_acce, axis=0)(ts)

		self.velocity = interp1d(self.tmp_ts, self.tmp_velocity, axis=0)(ts)

		ts = ts[:, np.newaxis]

		ori_R_gt = r
		ori_R = ori_R_gt

		glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
		glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)
		glob_acce -= np.array([0.0, 0.0, 9.80665])

		self.timestamps = ts
		self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
		self.orientations = ori_R.as_quat()  # [x, y, z, w]
		self.gt_pos = p_interp
		self.gt_ori = ori_R_gt.as_quat()
		self.targets = np.concatenate([p_interp, ori_R.as_quat()], axis=1)  # [x, y, z, w]
		self.acceleration = glob_acce
		self.gyroscope = glob_gyro
		return True

	def get_feature(self):
		"""Obtain feature information from the dataset

		Returns:
			self.features: [a_x, a_y, a_z, g_x, g_y, g_z]
		"""
		return self.features

	def get_accel(self):
		return self.acceleration

	def get_gyro(self):
		return self.gyroscope

	def get_target(self):
		return self.targets

	def get_data_valid(self):
		return self.data_valid

	def get_aux(self):
		return np.concatenate(
			[self.timestamps, self.orientations, self.gt_pos, self.gt_ori], axis=1
		)
	

class BasicSequenceData(object):
	def __init__(self, cfg, data_list, verbose=True, **kwargs):
		super(BasicSequenceData, self).__init__()
		self.window_size = int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])
		self.past_data_size = int(cfg['model_param']['past_time'] * cfg['data']['imu_freq'])
		self.future_data_size = int(cfg['model_param']['future_time'] * cfg['data']['imu_freq'])
		self.step_size = int(cfg['data']['imu_freq'] / cfg['data']['sample_freq'])
		self.seq_len = cfg['train']["seq_len"]

		self.index_map = []
		self.ts, self.orientations, self.gt_pos, self.gt_ori = [], [], [], []
		self.features, self.targets = [], []
		self.accel, self.gyro = [], []
		self.valid_t, self.valid_samples = [], []
		self.data_paths = []
		self.valid_continue_good_time = 0.1

		self.mode = kwargs.get("mode", "train")
		sum_t = 0
		win_dt = self.window_size / cfg['data']['imu_freq']
		self.valid_sum_t = 0
		self.valid_all_samples = 0
		max_v_norm = 4.0
		valid_i = 0
		for i in range(len(data_list)):
			seq = Sequence(
				data_list[i], 'data.h5', cfg['data']['imu_freq'], self.window_size, verbose=verbose
			)
			if seq.valid is False:
				continue
			accel = seq.get_accel()
			gyro = seq.get_gyro()
			feat = seq.get_feature()
			targ = seq.get_target()
			aux = seq.get_aux()
			sum_t += seq.sum_dur
			valid_samples = 0
			index_map = []
			step_size = self.step_size
			if self.mode in ["train", "val"] is False:
				for j in range(
						self.past_data_size,
						targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size,
						step_size):
					outlier = False
					for k in range(self.seq_len):
						index = j + k * self.window_size
						velocity = np.linalg.norm(targ[index] / win_dt)
						if velocity > max_v_norm:
							outlier = True
							break
					if outlier is False:
						index_map.append([valid_i, j])
						self.valid_all_samples += 1
						valid_samples += 1
			else:
				for j in range(
						self.past_data_size,
						targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size,
						step_size):
					index_map.append([valid_i, j])
					self.valid_all_samples += 1
					valid_samples += 1

			if len(index_map) > 0:
				self.data_paths.append(data_list[i])
				self.index_map.append(index_map)
				self.accel.append(accel)
				self.gyro.append(gyro)
				self.features.append(feat)
				self.targets.append(targ)
				self.ts.append(aux[:, 0])
				self.orientations.append(aux[:, 1:5])
				self.gt_pos.append(aux[:, 5:8])
				self.gt_ori.append(aux[:, 8:12])
				self.valid_samples.append(valid_samples)
				valid_i += 1
		if verbose:
			logging.info(f"---> datasets sum time {sum_t}")

	def get_data(self):
		return self.features, self.accel, self.gyro, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori

	def get_index_map(self):
		return self.index_map

	def get_merged_index_map(self):
		index_map = []
		for i in range(len(self.index_map)):
			index_map += self.index_map[i]
		return index_map