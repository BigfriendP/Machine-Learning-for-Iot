import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import time
import base64
import paho.mqtt.client as PahoMQTT
from utils import pad, get_mfccs, predict

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class SlowService:
	def __init__(self, clientID, topic):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect 
		self._paho_mqtt.on_message = self.myOnMessageReceived 
		
		self.topic = topic
		
		self.messageBroker = 'broker.emqx.io'  #'test.mosquitto.org'           

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()
		# subscribe for a topic
		self._paho_mqtt.subscribe(self.topic, 2)
	
	def stop (self):
		self._paho_mqtt.unsubscribe(self.topic)
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myPublish(self, topic, message):
		# publish a message with a certain topic
		#print("sent")
		self._paho_mqtt.publish(topic, message, 2) 

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		#print ("Connected to %s with result code: %d" % (self.messageBroker, rc))
		pass
		
	def myOnMessageReceived (self, paho_mqtt , userdata, msg): # define the preprocessing on the message received
		# A new message is received
		#print("received")
		message = json.loads(msg.payload)
		audio = base64.b64decode(message['e'][0]['audio'])
		audio = tf.convert_to_tensor(audio)
		audio, _ = tf.audio.decode_wav(audio)
		audio = tf.squeeze(audio, axis=1)

		frame_length = 640
		frame_step = 320
		num_spectrogram_bins = int(frame_length) // 2 + 1
		
		options = {'num_mel_bins': 40,
			'num_spectrogram_bins': num_spectrogram_bins,
			'lower_edge_hertz': 20,
			'upper_edge_hertz': 4000
			}
		sampling_rate = 16000

		audio = pad(audio, sampling_rate)
		
		linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(sample_rate = sampling_rate, **options)

		audio_mfcc = get_mfccs(audio,
					frame_length = frame_length,
					frame_step = frame_step,   
					linear_to_mel_weight_matrix = linear_to_mel_weight_matrix,
					num_coefficients = 10)
		
		audio_mfcc = tf.expand_dims(audio_mfcc, -1)
		audio_mfcc = tf.reshape(audio_mfcc, [1,49,10,1], name=None)

		pred = predict('kws_dscnn_True.tflite', audio_mfcc)

		message = {'n': 'audio', 'prediction': str(pred)} 
		message = json.dumps(message)
		time.sleep(0.1)
		self.myPublish("/predictions", message)
		
		
if __name__ == "__main__":
	service = SlowService("SlowService", '/unconfident')
	service.start()

	for i in range(120):
		time.sleep(1)
	service.stop()


	
		

