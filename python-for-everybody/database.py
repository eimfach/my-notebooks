import sqlite3
import os
import re

fileHandle = open(os.getcwd() + "/mbox-short.txt")
connection = sqlite3.connect("lectures.sqlite")

cursor = connection.cursor()
cursor.execute("DROP TABLE IF EXISTS EmailCount")
cursor.execute("CREATE TABLE EmailCount (email TEXT, count INTEGER)")

emails = re.findall("From:\s(\S+@\S+)", fileHandle.read())
emailHistogram = dict()

for email in emails:
    currentCount = emailHistogram.get(email, 0)
    emailHistogram[email] = currentCount + 1

for email,count in emailHistogram.items():
    cursor.execute("INSERT INTO EmailCount (email, count) VALUES (?, ?)", (email, count))

connection.commit()

# flow back
flowBackQuery = "SELECT * FROM EmailCount ORDER BY count DESC LIMIT 10"
for row in cursor.execute(flowBackQuery):
    print(str(row[0]), row[1])

cursor.close()
connection.close()