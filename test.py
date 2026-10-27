import requests

from applicant import applicant

url = "http://localhost:9696/predict"

response = requests.post(url, json=applicant).json()
print(response)

if response["default"]:
    print("High risk: likely to default")
else:
    print("Low risk: likely to repay")