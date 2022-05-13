import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import json
import cherrypy
import adafruit_dht
from board import D4
import base64
from simplePublisher import MyPublisher
import time

class AddModel(object):
    exposed=True
        
    def GET(self, *path, **query):
        pass
    
    def POST(self, *path, **query):
        
        #check if path and query are as expected

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        folder_path='./models/'
        
        #check if the models' folder exists, otherwise create it

        if os.path.isdir(folder_path) is False:
            os.mkdir(folder_path)

        #read the request's body

        body = cherrypy.request.body.read()
        body = json.loads(body)
        
        model = body.get('model')

        #check if model and name are correct

        if model is None:
            raise cherrypy.HTTPError(400, 'model missing')

        name = body.get('name')

        if name is None:
            raise cherrypy.HTTPError(400, 'model name missing')

        #decode the model 

        model_64bytes = base64.b64decode(model)
        
        #save the model

        with open(folder_path+name, 'wb') as f:
            f.write(model_64bytes)
		
    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


class ListModels(object):
    exposed = True

    def GET(self, *path, **query):
        
        #check if path and query are as expected

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        
        folder_path = "./models"
        
        #check if the models' folder exists

        if os.path.isdir(folder_path) is False:
            raise cherrypy.HTTPError(400, 'directory missing')

        #retrieve models' name, (without '.tflite')

        models = os.listdir(folder_path)
        models_name = []
        for model in models:
            models_name.append(model.split('.')[0])

        #create the json file with info for the client

        output = {'models': models_name}
        output_json = json.dumps(output)

        return output_json

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


class Predict(object):
    exposed=True

    def __init__(self):
        self.dht_device = adafruit_dht.DHT11(D4)
        
        #instantiate the mqtt publisher
        
        self.test = MyPublisher("Publisher")
        self.test.start()

    def GET(self, *path, **query):

        #check if path and query are as expected
        
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) != 3:
            raise cherrypy.HTTPError(400, 'Wrong query')

        #read info from the url

        model = query.get('model')
        tthres = np.float32(query.get('tthres'))
        hthres = np.float32(query.get('hthres'))

        model_path = './models/{}.tflite'.format(model)
        
        #check if the model exists

        if os.path.isfile(model_path) is False:
            raise cherrypy.HTTPError(400, 'model for the prediction missing')

        #load the tflite model

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        #define the window and the expected tensors

        window = np.zeros([1, 6, 2], dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)

        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)

        i = 0
        while True:

            #measure temp and hum and store the posix 

            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            posix = int((datetime.now()).timestamp())

            #create the first window

            if i < 6:
                window[0, i, 0] = np.float32(temperature)
                window[0, i, 1] = np.float32(humidity)
                i = i+1

            #predict next temp and hum and compare it with measuraments    

            else:
                expected[0] = np.float32(temperature)
                expected[1] = np.float32(humidity)

                curr_window = (window - MEAN) / STD
                interpreter.set_tensor(input_details[0]['index'], curr_window)
                interpreter.invoke()
                predicted = interpreter.get_tensor(output_details[0]['index'])

                abs_t_err = abs(expected[0]-predicted[0,0])
                abs_h_err = abs(expected[1]-predicted[0,1])
                
                #if MAEs are above the respective threshold send the alert

                if abs_t_err > tthres:
                    message = {
                        'bn': 'raspberrypi.local',
                        'e':[
                            {'n': 'Temperature', 'u': '°C', 't': posix, 'expected_val': float(expected[0])},
                            {'n': 'Temperature', 'u': '°C', 't': posix, 'predicted_val': float(predicted[0,0])}   
                        ]
                    }
                    message = json.dumps(message)
                    self.test.myPublish('/temperature_alert', message)

                if abs_h_err > hthres:
                    message = {
                        'bn': 'raspberrypi.local',
                        'e':[
                            {'n': 'Humidity', 'u': '%', 't': posix, 'expected_val': float(expected[1])},
                            {'n': 'Humidity', 'u': '%', 't': posix, 'predicted_val': float(predicted[0,1])}   
                        ]
                    }
                    message = json.dumps(message)
                    self.test.myPublish('/humidity_alert', message)
                
                #update the window

                window = np.roll(window, -1, axis = 1)
                
                window[0, 5, 0] = np.float32(temperature)
                window[0, 5, 1] = np.float32(humidity)

            time.sleep(1)
            
    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}

    cherrypy.tree.mount(AddModel(), '/add', conf)
    cherrypy.tree.mount(ListModels(), '/list', conf)
    cherrypy.tree.mount(Predict(), '/predict', conf)

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()