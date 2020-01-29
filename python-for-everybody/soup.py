import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

url = input("Enter Url: ")
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')

anchors = soup("a")

for anchor in anchors:
    print(anchor.get("href", None))