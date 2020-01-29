import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

htmlStr = urllib.request.urlopen("http://py4e-data.dr-chuck.net/comments_362873.html").read()
html = BeautifulSoup(htmlStr, "html.parser")

spanList = html("span")

# operations on a tag
#   print 'URL:',tag.get('href', None)
#   print 'Contents:',tag.contents[0]
#   print 'Attrs:',tag.attrs

comments = sum([int(span.contents[0]) for span in spanList ])

print(comments)