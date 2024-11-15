import requests

print('hai')

url = "https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot"


data = {
    "message": "what are the Features of Indian Constitution in 3 sentances"
}

response = requests.post(url, json=data)



print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.content}")

try:
    response_json = response.json()
    print(response_json)
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response")