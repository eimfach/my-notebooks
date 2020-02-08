import sqlite3
import json
import codecs

connection = sqlite3.connect('geodata.sqlite')
cursor = connection.cursor()

cursor.execute('SELECT * FROM Locations')
fhand = codecs.open('where.js', 'w', "utf-8")
fhand.write("myData = [\n")
count = 0
for row in cursor:
    # data = str(row[1])
    # not decoding the data from the database will result in a json parsing error later on
    data = str(row[1].decode())

    try:
        js = json.loads(data)
    except json.JSONDecodeError:
        print("Error parsing json.", str(row[0].decode()))
        continue

    if not('status' in js and js['status'] == 'OK'):
        print("Http response status not ok")
        continue

    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]
    if lat == 0 or lng == 0:
        print("lat & lng == 0")
        continue
    where = js['results'][0]['formatted_address']
    where = where.replace("'", "")
    try:
        print(where, lat, lng)

        count = count + 1
        if count > 1:
            fhand.write(",\n")
        output = "["+str(lat)+","+str(lng)+", '"+where+"']"
        fhand.write(output)
    except:
        continue

fhand.write("\n];\n")
cursor.close()
fhand.close()
print(count, "records written to where.js")
print("Open where.html to view the data in a browser")
