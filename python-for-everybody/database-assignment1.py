import sqlite3
import os
import re

fileHandle = open(os.getcwd() + "/mbox.txt")
orgs = re.findall("From:\s\S+@(\S+)", fileHandle.read())
orgHistogram = dict()

connection = sqlite3.connect("assignment-rge-1.sqlite")
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS Counts")
cursor.execute("CREATE TABLE Counts (org TEXT, count INTEGER)")

for org in orgs:
    orgHistogram[org] = orgHistogram.get(org, 0) + 1

for org,count in orgHistogram.items():
    cursor.execute("INSERT INTO Counts (org, count) VALUES (?, ?)", (org, count))

connection.commit()

cursor.close()
connection.close()