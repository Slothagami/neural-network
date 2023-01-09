import requests

req = requests.get("https://web.microsoftstream.com/1575a924-febe-463f-8376-e144578916f3", allow_redirects=True)
with open("vid", "wb") as file: file.write(req.content)
