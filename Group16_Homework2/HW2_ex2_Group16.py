import argparse 
from scipy import signal
import numpy as np
import os
import zlib
import tensorflow as tf
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model version')
args = parser.parse_args()

if args.version != 'a' and args.version != 'b' and args.version != 'c':
	print('ERROR -> INVALID INPUT: version must be a or b or c')
	quit()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
	origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
	fname='mini_speech_commands.zip',
	extract=True,
	cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

#create the tensors to split the dataset

train_txt = open("kws_train_split.txt", "r")
train_files = train_txt.read().splitlines()
train_files = tf.convert_to_tensor(train_files)
train_txt.close()

val_txt = open("kws_val_split.txt", "r")
val_files = val_txt.read().splitlines()
val_files = tf.convert_to_tensor(val_files)
val_txt.close()

test_txt = open("kws_test_split.txt", "r")
test_files = test_txt.read().splitlines()
test_files = tf.convert_to_tensor(test_files)
test_txt.close()

#create the tensor with labels

labels_txt = open("labels.txt", "r")
labels = labels_txt.read().translate({ord('['): None, ord(']'): None, ord("'"): None})
labels = labels.translate({ord(' '): None}).split(",")
labels = tf.convert_to_tensor(labels)
labels_txt.close()

#function for resampling

def resampling_func(audio, sampling_rate):        
	audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
	audio = audio.astype(np.float32)
	return audio

    
#function for weights + activations quantization

def representative_dataset_generator():
	for x, _ in train_ds.take(1000):
		yield [x]
    
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


#choosing preprocessing parameters based on the version to run

#params values for version a

if args.version == 'a': 
	options = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
		'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
		'num_coefficients': 10}
	sampling_rate = 16000 

#params for version b and c

else : 
	options = {'frame_length': 320, 'frame_step': 160, 'mfcc': True,
		'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 16,
		'num_coefficients': 10}
	sampling_rate = 8000
    
strides = [2, 1]


# Create train, validation and test datasets

print('datasets creation')

generator = SignalGenerator(labels, sampling_rate, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


print('model definition')

# set alpha for structured pruning

if args.version == 'a':
	alpha = 1
elif args.version == 'b':
	alpha = 0.7
else:
	alpha = 0.3
    

# DS-CNN 
model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[3,3], strides=strides, use_bias=False),
	tf.keras.layers.BatchNormalization(momentum=0.1),
	tf.keras.layers.ReLU(),
	tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
	tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1,1], strides=[1,1], use_bias=False),
	tf.keras.layers.BatchNormalization(momentum=0.1),
	tf.keras.layers.ReLU(),
	tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
	tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1,1], strides=[1,1], use_bias=False),
	tf.keras.layers.BatchNormalization(momentum=0.1),
	tf.keras.layers.ReLU(),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(units = 8)
	])

# Define loss, optimizer and metric
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


# Define the parameters for each version

if args.version == 'a':
	epochs = 25
    
elif args.version == 'b':
	epochs = 20
	pruning_params = {'pruning_schedule':
	tfmot.sparsity.keras.PolynomialDecay(
		initial_sparsity=0.40,
		final_sparsity=0.73,
		begin_step=3*len(train_ds),
		end_step=19*len(train_ds)
		)
	}
else:
	epochs = 25
	pruning_params = {'pruning_schedule':
	tfmot.sparsity.keras.PolynomialDecay(
		initial_sparsity=0.15,
		final_sparsity=0.35,
		begin_step=2*len(train_ds),
		end_step=25*len(train_ds)
		)
	}


if args.version != 'a':

	# training model with magnitude pruning
	
	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	model = prune_low_magnitude(model, **pruning_params)

	callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

	# Train the model
	input_shape = [1, 49, 10, 1]
	model.build(input_shape) 
	model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
	print('training model for', epochs ,'epochs')
	model.fit(train_ds, epochs=epochs,validation_data=val_ds, callbacks=callbacks, verbose=2)
	print(model.summary())

	# Strip the model after training
	model = tfmot.sparsity.keras.strip_pruning(model)
	
else:
	
	# Make the model training without magnitude pruning 	
	
	model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
	print('training model for', epochs ,'epochs')
	model.fit(train_ds, epochs=epochs,validation_data=val_ds, verbose=2)
	print(model.summary())
	
	
print('Quantization')

# Converting the model to a tfLite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 


if args.version == 'b':
    
	# weights+activations quantization
        
	converter.representative_dataset = representative_dataset_generator

elif args.version == 'c':

	# weights-only quantization

	converter.target_spec.supported_types = [tf.float16]
    
tflite_model = converter.convert()


print('save and evaluation of tflite model')

if not os.path.exists('./models/'):
	os.makedirs('./models/')

model_dir = os.path.join('.', 'models', 'Group16_kws_{}.tflite.zlib'.format(args.version))
with open(model_dir, 'wb') as fp:    
    
	tflite_compressed = zlib.compress(tflite_model)
	fp.write(tflite_compressed)

# Size of the final tflite.zlib model

print('Model size version {}: {:.2f}kB'.format(args.version, os.path.getsize(model_dir)/1024))


# Evaluation of the tflite model 
    
# decompress the zlib model

f = open(model_dir, 'rb')
decompressed_model = zlib.decompress(f.read())
    
interpreter = tf.lite.Interpreter(model_content=decompressed_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
dataset = test_ds.unbatch().batch(1)

outputs = []
labels = []
    
for data in dataset:
	my_input = np.array(data[0], dtype = np.float32)
	label = np.array(data[1], dtype = np.float32)
	labels.append(label)

	interpreter.set_tensor(input_details[0]['index'], my_input)
	interpreter.invoke()
	my_output = interpreter.get_tensor(output_details[0]['index'])
	outputs.append(my_output[0])
        
outputs = np.array(outputs)
labels = np.squeeze(np.array(labels))
    
acc = sum(np.equal(labels, np.argmax(outputs, axis=1)))/len(outputs)
print('Accuracy of model {} = {:.3f}'.format(args.version, acc))
                
    
    

