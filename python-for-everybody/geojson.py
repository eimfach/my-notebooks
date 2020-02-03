import urllib.request, urllib.parse, urllib.request
import ssl
import json

serviceUrl = "https://maps.googleapis.com/maps/api/geocode/json?"

key = "key=AIzaSyC9ZU-fi5qltBUpC5J_P6j5CdugL7x3SqM"

while True:
    address = input('Enter location:')
    if len(address) == 0:
        break

    url = serviceUrl + urllib.parse.urlencode({"address": address})
    url = url + "&" + key

    sslContext = ssl.SSLContext()
    resp = urllib.request.urlopen(url, context=sslContext)

    data = resp.read().decode()
    print("Response: [" + str(len(data)) + "]", resp.getheaders())

    try:
        js = json.loads(data)
    except:
        js = None

    if not js:
        print("Error parsing response as JSON")
        break

    if js["status"] != "OK":
        print("API responded not OK")
        break

    print(json.dumps(js, indent=4))

    firstResult = js["results"][0]
    formattedAddress = firstResult["formatted_address"]
    geometry = firstResult["geometry"]
    lat = geometry["location"]["lat"]
    lng = geometry["location"]["lng"]
    print(formattedAddress)
    print(lat, lng)