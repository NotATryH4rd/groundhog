import requests

url = "https://groundhog.pythonanywhere.com/predict"

data = {
    "Temperature": 25,
    "Moisture": 40,
    "ph": 6.5,
    "EC": 0.8
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("✅ yayyyyyyy:", response.json())
else:
    print("❌ screw u:", response.status_code, response.text)
