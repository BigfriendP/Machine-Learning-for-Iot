import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import time
import base64
import paho.mqtt.client as PahoMQTT
from utils import SignalGenerator, load_eval



seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class FastClient():
	
	def __init__(self, clientID, topic, confident_outputs, labels, n_unconf):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect
		self._paho_mqtt.on_message = self.myOnMessageReceived
		
		self.topic = topic
		self.messageBroker = 'broker.emqx.io'  #'test.mosquitto.org'           

		self.confident_outputs = confident_outputs
		self.n_unconf = n_unconf
		self.received = 0
		self.labels = labels

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()
		self._paho_mqtt.subscribe(self.topic, 2)

	def stop (self):
		self._paho_mqtt.unsubscribe(self.topic)
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myPublish(self, topic, msg):
		# publish a message with a certain topic
		#print("sent") 
		self._paho_mqtt.publish(topic, msg, 2)

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		#print ("Connected to %s with result code: %d" % (self.messageBroker, rc))
		pass

	def collaborative_accuracy(self):
		outputs = np.squeeze(self.confident_outputs)
		labels = np.squeeze(np.array(self.labels))
		accuracy = sum(np.equal(labels, np.argmax(outputs, axis = 1)))/len(outputs)
		print('Accuracy: {:.3f}%'.format(accuracy*100))

	def myOnMessageReceived (self, paho_mqtt , userdata, msg):
		# A new message is received
		#print("received")
		self.received = self.received + 1 
		message = json.loads(msg.payload)
		output = json.loads(message['prediction'])
		
		self.confident_outputs.append(np.array(output))
		if self.received == self.n_unconf:
			self.collaborative_accuracy()


if __name__ == "__main__":

	#tensor to create the test set

	test_txt = open("kws_test_split.txt", "r")
	test_files = test_txt.read().splitlines()
	test_files = tf.convert_to_tensor(test_files)
	test_txt.close()
	num_samples_test = test_files.shape

	#create the tensor with labels

	labels_txt = open("labels.txt", "r")
	labels = labels_txt.read().translate({ord('['): None, ord(']'): None, ord("'"): None})
	labels = labels.translate({ord(' '): None}).split(",")
	labels = tf.convert_to_tensor(labels)
	labels_txt.close()

	options = {'frame_length': 320, 'frame_step': 160, 'mfcc': True,
					'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 16,
					'num_coefficients': 10}
	sampling_rate = 8000

	#create test dataset 

	generator = SignalGenerator(labels, sampling_rate = sampling_rate, **options)
	test_ds = generator.make_dataset(test_files, False)
	model_directory = 'kws_dscnn_True.tflite'

	#evaluate with tflite model

	labels, conf_outputs, unconf_idx = load_eval(model_directory, test_ds)
	n_unconf = len(unconf_idx)
	#print('number of unconfident predictions: ', n_unconf)
	comunication_cost = 0

	client = FastClient("FastClient", '/predictions', conf_outputs, labels, n_unconf)
	client.start()

	for idx in unconf_idx:
		with open((test_files.numpy())[idx], 'rb') as f:
			audio = f.read()
		audio_b64 = base64.b64encode(audio)
		audio_str = audio_b64.decode()

		posix = int((datetime.now()).timestamp())

		message = {
			'bn': 'raspberrypi.local',
			'e': [
				{'n': 'audio '+ str(idx), 'u': '', 't': posix, 'audio': audio_str}
			]
		}

		message = json.dumps(message)
		comunication_cost += len(message)
		time.sleep(0.3)
		client.myPublish("/unconfident", message)

	print('Communication Cost: {:.3f} MB'.format(comunication_cost/1e+6))

	for i in range(120):
		time.sleep(1)
	#print('Communication Cost: {:.3f} MB'.format(comunication_cost/1e+6))
	client.stop()
	