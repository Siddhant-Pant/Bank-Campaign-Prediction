import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json = {'age':30, 'salary':60000, 'loan':0})

print(r.json())