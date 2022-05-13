import requests
import json
import base64

#add requests

print("Add model requests...")
model1_name = './cnn_2.tflite'
model2_name = './mlp_2.tflite'

#load and convert the first model

with open(model1_name, 'rb') as f:
	model1 = f.read()
model1_64bytes = base64.b64encode(model1)

#load and convert the second model

with open (model2_name, 'rb') as f1:
	model2 = f1.read()
model2_64bytes = base64.b64encode(model2)

#define the url for the add requests

url = 'http://raspberrypi.local:8080/add'

#define body of the first add request

body = {'model': str(model1_64bytes.decode('utf-8')),
		'name': model1_name}

r = requests.post(url , json=body)

if r.status_code == 200:
	print(r.status_code)
else:
	print('Error:', r.status_code)

#define body of the second add request

body = {'model': str(model2_64bytes.decode('utf-8')),
		'name': model2_name}

r = requests.post(url , json=body)

if r.status_code == 200:
	print(r.status_code)
else:
	print('Error:', r.status_code)


#list request 

print("List models request")

url = 'http://raspberrypi.local:8080/list'

#define the url for the list requests

r = requests.get(url)

if r.status_code == 200:
    body = r.json()
    print(r.status_code)
    print(body['models'])
else:
    print('Error:', r.status_code)

#predict request

print("Prediction request")

#define the url for the prediction request
#change cnn_2 with mlp_2 to use the mlp

url = 'http://raspberrypi.local:8080/predict/?model=cnn_2&tthres=0.1&hthres=0.2'

r = requests.get(url)

if r.status_code == 200:
    print(r.status_code)
else:
    print('Error:', r.status_code)