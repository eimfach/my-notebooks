import json
import urllib.request, urllib.parse

def fixture():
    data = '''{
        "name": "Chuck",
        "phone": {
            "type": "tel",
            "value": "+1 123"
        },
        "email": {
            "show": false
        }
    }'''

    info = json.loads(data)
    print("Name", info["name"])
    print("Show E-Mail:", info["email"])
    print(type(info["email"]))

def fromUrl():
    # url = input("Enter URL:")
    url = "http://py4e-data.dr-chuck.net/comments_362876.json"
    resp = urllib.request.urlopen(url)
    data = resp.read().decode()
    js = json.loads(data)

    comments = js["comments"]

    commentCount = sum([int(comment["count"]) for comment in comments])
    print(commentCount)
    
def assignment():
    url = "http://py4e-data.dr-chuck.net/json?"

    address = urllib.parse.urlencode({"address": "Jordan University of Science and Technology"})
    
    url = url + address + "&key=42"
    print("Request to url: ", url)

    resp = urllib.request.urlopen(url)
    data = resp.read().decode()
    js = json.loads(data)
    print(js["results"][0]["place_id"])

assignment()