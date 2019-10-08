import json
import os
import glob
import numpy as np
import csv
import tensorflow as tf
import random

TRAINING_RATIO = 0.8
TESTING_RATIO = 0.2

ACCELEROMETER_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_accelerometer_clips"
GYROSCOPE_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_gyroscope_clips"
# GYROSCOPE_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_accelerometer_clips"
TF_RECORD_PATH = r'A:\Research\Accelerometer\AccelerometerSpeechRecognition\STFNets\speech'

# SPECTURM_SAMPLE_NUM = 25
SPECTURM_SAMPLE_NUM = 10

LABELS = range(0, 10)

def load_data_under_dir(data_dir):
	filenames_under_dir = [name for name in glob.glob("{}\*.txt".format(data_dir))]
	data = []
	for filename in filenames_under_dir:
		with open(filename, "r") as file:
			data.append(json.load(file))
	return data


accelerometer_data = load_data_under_dir(ACCELEROMETER_FILE_PATH)
gyroscope_data = load_data_under_dir(GYROSCOPE_FILE_PATH)


def preprocess_one_label(accelerometer_data, gyroscope_data, label):
	X = []
	one_hot_label = np.zeros(len(LABELS))
	one_hot_label[label] = 1

	for (acc_clip, gyro_clip) in zip(accelerometer_data, gyroscope_data):
		features = [] 
		padding_element = [[0,0,0]] * 131
		acc_clip -= np.array(acc_clip).mean(axis=0, keepdims=True)
		# print("acc_clip.shape = {}".format(acc_clip.shape))
		acc_clip = np.concatenate((padding_element, acc_clip, padding_element))
		gyro_clip -= np.array(gyro_clip).mean(axis=0, keepdims=True)
		gyro_clip = np.concatenate((padding_element, gyro_clip, padding_element))

		for (a, g) in zip(acc_clip, gyro_clip):
			features.append(np.concatenate((a, g)))
		
		features = np.array(features).flatten()
		
		print("features shape: {}".format(np.array(features).shape))

		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(float_list=tf.train.FloatList(value=one_hot_label)),
			'example': tf.train.Feature(float_list=tf.train.FloatList(value=features))
		}))

		X.append(example)
	return X

examples = []
for label in LABELS:
	print("label: {}".format(label))
	X = preprocess_one_label(accelerometer_data[label], gyroscope_data[label], label)
	examples += X

print("example number = {}".format(len(examples)))

training_examples = examples[:int(len(examples)*TRAINING_RATIO)]
testing_examples = examples[:int(len(examples)*TESTING_RATIO)]

print("training_examples len = {}, testing_examples len = {}".format(len(training_examples), len(testing_examples)))

writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'train.tfrecord'))

for e in training_examples:
	writer.write(e.SerializeToString())

writer.close()

writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'eval.tfrecord'))

for e in testing_examples:
	writer.write(e.SerializeToString())

writer.close()

	