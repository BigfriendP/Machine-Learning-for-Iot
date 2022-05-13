import numpy as np
import tensorflow as tf
import os
from scipy import signal

#function for resampling

def resampling_func(audio, sampling_rate):        
	audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
	audio = audio.astype(np.float32)
	return audio

#define the class for reading and preprocessing of wav files

class SignalGenerator:
	def __init__(self, labels, sampling_rate, frame_length, frame_step,
		num_mel_bins=None, lower_frequency=None, upper_frequency=None,
		num_coefficients=None, mfcc=False):
		
		self.labels = labels
		self.sampling_rate = sampling_rate
		self.frame_length = frame_length
		self.frame_step = frame_step
		self.num_mel_bins = num_mel_bins
		self.lower_frequency = lower_frequency
		self.upper_frequency = upper_frequency
		self.num_coefficients = num_coefficients
		num_spectrogram_bins = (frame_length) // 2 + 1

		if mfcc is True:
			self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
				self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
				self.lower_frequency, self.upper_frequency)
			self.preprocess = self.preprocess_with_mfcc
		else:
			self.preprocess = self.preprocess_with_stft

	def read(self, file_path):
		parts = tf.strings.split(file_path, os.path.sep)
		label = parts[-2]
		label_id = tf.argmax(label == self.labels)
		audio_binary = tf.io.read_file(file_path)
		audio, _ = tf.audio.decode_wav(audio_binary)
		audio = tf.squeeze(audio, axis=1)
		#make resampling
		if(self.sampling_rate != 16000):  
			audio = tf.numpy_function(resampling_func, [audio, self.sampling_rate], tf.float32)

		return audio, label_id

	def pad(self, audio):
		zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
		audio = tf.concat([audio, zero_padding], 0)
		audio.set_shape([self.sampling_rate])

		return audio

	def get_spectrogram(self, audio):
		stft = tf.signal.stft(audio, frame_length=self.frame_length,
			frame_step=self.frame_step, fft_length=self.frame_length)
		spectrogram = tf.abs(stft)

		return spectrogram

	def get_mfccs(self, spectrogram):
		mel_spectrogram = tf.tensordot(spectrogram,
			self.linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[..., :self.num_coefficients]

		return mfccs

	def preprocess_with_stft(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		spectrogram = tf.expand_dims(spectrogram, -1)
		spectrogram = tf.image.resize(spectrogram, [32, 32])

		return spectrogram, label

	def preprocess_with_mfcc(self, file_path):
		audio, label = self.read(file_path)
		audio = self.pad(audio)
		spectrogram = self.get_spectrogram(audio)
		mfccs = self.get_mfccs(spectrogram)
		mfccs = tf.expand_dims(mfccs, -1)

		return mfccs, label

	def make_dataset(self, files, train):
		ds = tf.data.Dataset.from_tensor_slices(files)
		ds = ds.map(self.preprocess, num_parallel_calls=4)
		ds = ds.batch(32)
		ds = ds.cache()
		if train is True:
			ds = ds.shuffle(100, reshuffle_each_iteration=True)

		return ds


#function to pad the signal

def pad(audio, sample_rate):
	zero_padding = tf.zeros([sample_rate] - tf.shape(audio), dtype=tf.float32)
	audio = tf.concat([audio, zero_padding], 0)
	audio.set_shape([sample_rate])

	return audio

#function to get the MFCC of an audio

def get_mfccs(audio, frame_length, frame_step, linear_to_mel_weight_matrix, num_coefficients):
	stft = tf.signal.stft(audio, frame_length=frame_length,
				frame_step=frame_step, fft_length=frame_length)
	spectrogram = tf.abs(stft)

	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
	mfccs = mfccs[..., : num_coefficients]
	return mfccs


	


#succes checker: to check if the prediction is confident or not

def success_checker(pred, thres):
	x = np.exp(pred - np.max(pred)) 
	softmax_pred = np.squeeze(x/x.sum())
	out_prob = np.sort(softmax_pred)[-1]
	#check if the prediction is not confident
	if out_prob < thres: 
		return False
	return True


#function to load and evaluate the model

def load_eval(model_path, dataset):
	#decompress the zlib model
	f = open(model_path, 'rb')
	model = f.read()

	interpreter = tf.lite.Interpreter(model_content = model)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	dataset = dataset.unbatch().batch(1)

	conf_outputs = []
	labels = []
	unconf_idx = []
	unconf_labels = []

	for idx, data in enumerate(dataset):
		inp = np.array(data[0], dtype = np.float32)
		label = np.array(data[1], dtype = np.float32)

		interpreter.set_tensor(input_details[0]['index'],inp)
		interpreter.invoke()
		out = interpreter.get_tensor(output_details[0]['index'])

		if success_checker(out, 0.45):
			conf_outputs.append(np.squeeze(out))
			labels.append(int(label))
		else:
			unconf_idx.append(idx)
			unconf_labels.append(int(label))

	labels = labels + unconf_labels
	if len(unconf_idx) == 0:
		accuracy = sum(np.equal(labels, np.argmax(conf_outputs, axis = 1)))/len(conf_outputs)
		print('Accuracy: {:.2f}%'.format(accuracy*100))
	
	return labels, conf_outputs, unconf_idx

#function to make prediction on a mfcc

def predict(model_path, audio_mfcc):
	f = open(model_path, 'rb')
	model = f.read() 
	
	interpreter = tf.lite.Interpreter(model_content = model)
	interpreter.allocate_tensors()
	
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	interpreter.set_tensor(input_details[0]['index'], audio_mfcc)
	interpreter.invoke()

	out = interpreter.get_tensor(output_details[0]['index'])
	out= np.squeeze(np.array(out))
	
	return list(out)

