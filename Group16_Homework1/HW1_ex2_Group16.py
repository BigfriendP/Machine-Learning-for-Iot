import time
import os
import tensorflow as tf
import numpy as np
from scipy import signal
from subprocess import Popen

Popen('sudo sh -c "echo performance >'  
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',    
      shell=True).wait() 


directory = "yes_no"

#MFCC_slow parameters

#STFT parameters

l = 16*1e-3
s = 8*1e-3

#MFCC parameters

num_mel_bins = 40
num_mfccs = 10 
lower_frequency = 20
upper_frequency = 4000

def calculate_spectrogram(tf_audio, rate, frame_length, frame_step):

	samples_in_window = (rate*frame_length).astype(np.int32)
	samples_step = (rate*frame_step).astype(np.int32)

	stft = tf.signal.stft(tf_audio, frame_length = samples_in_window, frame_step=samples_step, fft_length=samples_in_window)

	spectrogram = tf.abs(stft)
	return spectrogram

def calculate_MFCC(spectrogram, linear_to_mel_weight_matrix, num_mfccs):

	#num_spectrogram_bins = spectrogram.shape[-1]
	#linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency, upper_frequency)

	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
	mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]

	return mfccs

count = 0
cumulative_time = 0

mfccs_slow = []

for filename in os.listdir(directory):

	start_time = time.time()

	if filename.endswith(".wav"):

		audio = tf.io.read_file(os.path.join(directory, filename))

		tf_audio, rate = tf.audio.decode_wav(audio)
		tf_audio = tf.squeeze(tf_audio, 1)
		rate = rate.numpy()

		spectrogram_slow = calculate_spectrogram(tf_audio, rate, l, s)

		#if filename == os.listdir(directory)[0]:
		if count == 0:
			print(tf_audio.shape)
			print(spectrogram_slow.shape)
			num_spectrogram_bins = spectrogram_slow.shape[-1]
			linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, rate, lower_frequency, upper_frequency)
			print(linear_to_mel_weight_matrix.shape)

		mfcc_slow = calculate_MFCC(spectrogram_slow, linear_to_mel_weight_matrix, num_mfccs)

		count+=1
		cumulative_time+=time.time()-start_time

		mfccs_slow.append(mfcc_slow)

	else:
		continue

avg_time_slow = cumulative_time/count
print("the average time of MFCC_slow is: ", avg_time_slow)

#MFCC_fast parameters


#resampling parameters

up = 1
down = 4

#STFT parameters

l_fast = 16*1e-3
s_fast = 8*1e-3

#MFCC parameters

num_mel_bins_f = 32
num_mfccs_f = 10 
lower_frequency_f = 20
upper_frequency_f = 2000


count = 0
cumulative_time = 0

mfccs_fast = []



for filename in os.listdir(directory):

	start_time = time.time()

	if filename.endswith(".wav"):

		audio = tf.io.read_file(os.path.join(directory, filename))
		
		tf_audio, rate = tf.audio.decode_wav(audio)
		tf_audio = tf.squeeze(tf_audio, 1)

		tf_audio = signal.resample_poly(tf_audio,up,down)
		tf_audio = tf_audio.astype(np.float32)
		rate = (rate.numpy()*up)/down

		spectrogram_fast = calculate_spectrogram(tf_audio, rate, l_fast, s_fast)

		#if filename == os.listdir(directory)[0]:
		if count == 0:
			print(tf_audio.shape)
			print(spectrogram_fast.shape)
			num_spectrogram_bins_f = spectrogram_fast.shape[-1]
			linear_to_mel_weight_matrix_f = tf.signal.linear_to_mel_weight_matrix(num_mel_bins_f, num_spectrogram_bins_f, rate, lower_frequency_f, upper_frequency_f)
			print(linear_to_mel_weight_matrix_f.shape)
		mfcc_fast = calculate_MFCC(spectrogram_fast, linear_to_mel_weight_matrix_f, num_mfccs_f)

		#mfcc_fast = tf.pad(mfcc_fast, paddings, "SYMMETRIC")

		count+=1
		cumulative_time+=time.time()-start_time

		mfccs_fast.append(mfcc_fast)

	else:
		continue

avg_time_fast = cumulative_time/count
print("the average time of MfCC_fast is: ", avg_time_fast)

'''print("the shapes of mfccs_slow are: ",len(mfccs_slow[0]))
print(len(mfccs_fast[0]),"\n")

print("the shapes of mfccs_slow are: ",len(mfccs_slow[0][0]))
print(len(mfccs_fast[0][0]))'''

snr_list = []

for i in range(len(mfccs_slow)):

	slow = np.array(mfccs_slow[i])
	fast = np.array(mfccs_fast[i])

	snr = 20*np.log10(np.linalg.norm(slow)/np.linalg.norm(slow-fast+1e-6))

	snr_list.append(snr)

print("The average SNR is: ", sum(snr_list)/len(snr_list))










