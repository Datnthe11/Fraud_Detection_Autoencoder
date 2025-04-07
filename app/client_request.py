import requests

import json 
data = {"features": [-1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698,
                   0.363787, 0.090794, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115,
                   0.133558, -0.021053, 0.005824, 0, 0, 0, 0, 0, 0, 0, 0, 0,0.005824]}
url = "http://127.0.0.1:8888/predict/"
data = json.dumps(data)

response = requests.post(url,data)
print(response.json())