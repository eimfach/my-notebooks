import urllib.request, urllib.parse, urllib.error
import sqlite3
import json
import time
import ssl

# Google API (requires API key)
# serviceurl = "http://maps.googleapis.com/maps/api/geocode/json?"
# If you are in China this URL might work (with key):
# serviceurl = "http://maps.google.cn/maps/api/geocode/json?"

serviceurl = "http://python-data.dr-chuck.net/geojson?"


# Deal with SSL certificate anomalies Python > 2.7
# scontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
scontext = None

connection = sqlite3.connect('geodata.sqlite')
current = connection.cursor()

current.execute('''
CREATE TABLE IF NOT EXISTS Locations (address TEXT, geodata TEXT)''')

locationsToLookup = open("where.data")
count = 0
for location in locationsToLookup:
    if count > 200:
        print('Retrieved 200 locations, restart to retrieve more')
        break
    address = location.strip()

    current.execute(
        "SELECT geodata FROM Locations WHERE address= ?", (memoryview(bytes(address, 'utf-8')), ))

    try:
        data = current.fetchone()[0]
        print("Found in database ", address)
        continue
    except:
        pass

    print('Resolving', address)
    url = serviceurl + \
        urllib.parse.urlencode({"sensor": "false", "address": address})
    print('Retrieving', url)
    uh = urllib.request.urlopen(url, context=scontext)
    data = uh.read().decode()
    print('Retrieved', len(data), 'characters', data[:20].replace('\n', ' '))
    count = count + 1
    try:
        js = json.loads(str(data))
        # print js  # We print in case unicode causes an error
    except:
        continue

    if 'status' not in js or (js['status'] != 'OK' and js['status'] != 'ZERO_RESULTS'):
        print('==== Failure To Retrieve ====')
        print(data)
        continue

    current.execute('''INSERT INTO Locations (address, geodata) 
            VALUES ( ?, ? )''', (memoryview(bytes(address, 'utf-8')), memoryview(bytes(data, 'utf-8'))))
    connection.commit()
    if count % 10 == 0:
        print('Pausing for a bit...')
        # time.sleep(5)

print("Run geodump.py to read the data from the database so you can visualize it on a map.")
