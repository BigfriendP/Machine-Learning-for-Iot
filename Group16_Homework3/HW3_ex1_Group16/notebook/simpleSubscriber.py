import paho.mqtt.client as PahoMQTT
import json
from datetime import datetime

class MySubscriber:
	def __init__(self, clientID, topic1, topic2):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect 
		self._paho_mqtt.on_message = self.myOnMessageReceived 
		
		self.topic1 = topic1
		self.topic2 = topic2
		self.messageBroker = 'test.mosquitto.org'

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()
		# subscribe for a topic
		
		self._paho_mqtt.subscribe(self.topic1, 2)
		self._paho_mqtt.subscribe(self.topic2, 2)
	
	def stop (self):
		self._paho_mqtt.unsubscribe(self.topic1)
		self._paho_mqtt.unsubscribe(self.topic2)
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.messageBroker, rc))
		
	def myOnMessageReceived (self, paho_mqtt , userdata, msg): # define the preprocessing on the message received
		# A new message is received
		message = json.loads(msg.payload)
		n = message["e"][0]["n"]
		u = message["e"][0]["u"]
		t = message["e"][0]["t"]
		actual_v = message["e"][0]["expected_val"]
		predicted_v = message["e"][1]["predicted_val"]

		date_time = str(datetime.fromtimestamp(t))

		print ("(" + date_time + ") " + n + " Alert: " + "Predicted={:.2f}".format(predicted_v) + u + " Actual={}".format(actual_v) + u)
