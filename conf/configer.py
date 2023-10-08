"""
Loading config files and building model
"""

import yaml
import os
import re
import logging
from datetime import datetime


# General config
def LoadConfig(path):
	""" Loads config file.
    Args:
        path (str): path to config file
    """
	# Load configuration from file itself
	with open(path, 'r') as f:
		cfg = yaml.load(f, Loader=yaml.Loader)

	return cfg


def UpdateRecursive(dict1, dict2):
	""" Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
	for k, v in dict2.items():
		if k not in dict1:
			dict1[k] = dict()
		if isinstance(v, dict):
			UpdateRecursive(dict1[k], v)
		else:
			dict1[k] = v


def GetTimeNow():
	now = datetime.now()
	# 提取年、月、日、小时和分钟
	year = now.year
	month = now.month
	day = now.day
	hour = now.hour
	minute = now.minute
	return year, month, day, hour, minute


def tryint(s):
	try:
		return int(s)
	except ValueError:
		return s


def str2int(v_str):
	return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def GetBestModel(path):
	names = sorted(os.listdir(path + "/"), key=str2int)
	files = []
	for name in names:
		if os.path.isfile(os.path.join(os.path.abspath(path), name)):
			files.append(name)
	# files.sort()
	model = os.path.join(os.path.abspath(path), files[-1])
	logging.info(f"---> load model: {model}")
	return model
