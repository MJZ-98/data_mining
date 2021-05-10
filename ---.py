import base64
import requests
#def predict(self, filenamepath):
filenamepath='D://Pic//apple.jpg'
global ignoreParams
ignoreParams=['人物特写','手']
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=UYlWIogoprDubrVlMHg9DzoE&client_secret=TGaLNXP9CHqVkmhLXQd6h9bOGqN37kTm'
response = requests.get(host)
if response:
    access_token = response.json()['access_token']
request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
f = open(filenamepath, 'rb')
img = base64.b64encode(f.read())
params = {"image": img}
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    dict = response.json()['result']
    print(dict)
    for i in dict:
        if i['keyword'] not in ignoreParams:
            print(i['keyword'])