import argparse
import os
import time
import pandas as pd
import tensorflow as tf
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to input csv file')
parser.add_argument('--output', type=str, help='Path to output tfrecord file')
parser.add_argument('--normalize', default=False, action = 'store_true', help='Whether to normalize the input data')

args = parser.parse_args()


T_min = 0
T_max = 50
H_min = 20
H_max = 90


#read the input file and store it in a pandas dataframe
 
try:
	input_dataset = pd.read_csv(args.input, header=None, names = ['date','time','temperature','humidity'])
except FileNotFoundError:
	msg = "Sorry, the file "+ args.input + " does not exist."
	print(msg)
	sys.exit()


#change the columns in datetime, temperature and humidity

input_dataset['datetime'] = input_dataset['date']+'/'+input_dataset['time']
input_dataset.drop(columns = ['date', 'time'], inplace = True)
input_dataset = input_dataset.reindex(columns = ['datetime', 'temperature', 'humidity'])
input_dataset['datetime'] = pd.to_datetime(input_dataset['datetime'])


#normalize temperature and humidity value if required

if args.normalize:
	input_dataset['temperature'] = (input_dataset['temperature'] - T_min)/(T_max-T_min)
	input_dataset['humidity'] = (input_dataset['humidity'] - H_min)/(H_max-H_min)
		

#write the tfrecord file 

with tf.io.TFRecordWriter(args.output) as writer: 
	for i in input_dataset.index:
        
		posix = int(time.mktime((input_dataset['datetime'][i]).timetuple()))
        
		posix_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[posix]))
        
		if args.normalize == False:
			temp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[input_dataset['temperature'][i]])) 
			hum_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[input_dataset['humidity'][i]])) 
            
		elif args.normalize == True:
			temp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[input_dataset['temperature'][i]])) 
			hum_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[input_dataset['humidity'][i]])) 
 
 
		mapping = {'posix': posix_feature, 'temp': temp_feature, 'hum'  : hum_feature} 
 
		example = tf.train.Example(features=tf.train.Features(feature=mapping)) 
        
		writer.write(example.SerializeToString()) 


#print the size of the tfrecord in bytes  

print("the size of the generated tfrecord is: ",os.path.getsize(args.output), "bytes")



