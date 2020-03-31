import xml.etree.ElementTree as ET
import urllib.request, urllib.parse, urllib.error

def xlm_sample():
    data = '''
    <person>
    <name>Sally</name>
    <email show="yes" />
    </person>
    '''

    invalidData = '''
    <person>
    <name>Sally</name>
    <email show="yes" />
    </person
    '''

    try:
        tree = ET.fromstring(data)
        print("Name:", tree.find("name").text)
        print("Show E-Mail Attr:", tree.find("email").get("show"))

        tree2 = ET.fromstring(invalidData)
    except:
        print("------")
        print("Invalid xml!")

def parse_xml_from_url():
    url = input("Enter Url:")
    resp = urllib.request.urlopen(url).read().decode()
    tree = ET.fromstring(resp)
    return sum([extractCommentCount(num) for num in tree.findall("comments/comment")])

def extractCommentCount(comment):
    return int(comment.find("count").text)

print(parse_xml_from_url())