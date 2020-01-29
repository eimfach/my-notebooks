import urllib.request, urllib.parse, urllib.error

fhand = urllib.request.urlopen("http://data.pr4e.org/romeo.txt")
# Similiar to file handling, awesome !
for line in fhand:
    # line is a byte array
    print(line.decode().strip())