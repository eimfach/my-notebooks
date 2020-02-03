import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

url = "http://py4e-data.dr-chuck.net/known_by_Jadon.html"
name = None

# operations on a tag
#   print 'URL:',tag.get('href', None)
#   print 'Contents:',tag.contents[0]
#   print 'Attrs:',tag.attrs

for step in range(7):
    if url is not None:
        print("Open up new link: " + url)
        htmlStr = urllib.request.urlopen(url).read()
        html = BeautifulSoup(htmlStr, "html.parser")

        anchors = html("a")
        thirdAnchor = list(anchors)[17]
        url = thirdAnchor.get("href", None)
        name = thirdAnchor.contents[0]

    else:
        print("Error while retrieving anchor href")

print(name)