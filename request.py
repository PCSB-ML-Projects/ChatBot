import requests
import json

# url = "http://localhost:7071/api/xeniachatbot/ask"
url = "http://localhost:5000/ask"

headers = {
    'Content-Type': 'application/json'
}

data = {
    "query": "explain xenalytics event"
}

# data = {
#     "content": "These discrepancies not only disrupt my ability to track and manage my finances accurately but also raise concerns about the reliability and integrity of the online banking system."
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.json())

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)
    res = response.json()
    print(res)
except Exception as e:
    print(e)