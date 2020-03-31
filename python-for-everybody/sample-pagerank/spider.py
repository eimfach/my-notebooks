import sqlite3
import urllib.error
import ssl
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

conn = sqlite3.connect('spider.sqlite')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS Pages
    (id INTEGER PRIMARY KEY, url TEXT UNIQUE, html TEXT,
     error INTEGER, old_rank REAL, new_rank REAL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Links
    (from_id INTEGER, to_id INTEGER)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Webs (url TEXT UNIQUE)''')

def selectRandomPage(cursor):
    cursor.execute('SELECT id,url FROM Pages WHERE html is NULL and error is NULL ORDER BY RANDOM() LIMIT 1')
    return cursor.fetchone()

# Randomly pick one page which did not retrieve yet
row = selectRandomPage(cursor)
if row is None:
    starturl = input('Enter web url or enter: ')
    if ( len(starturl) < 1 ) : starturl = 'http://www.dr-chuck.com/'
    if ( starturl.endswith('/') ) : starturl = starturl[:-1]
    web = starturl
    if ( starturl.endswith('.htm') or starturl.endswith('.html') ) :
        pos = starturl.rfind('/')
        web = starturl[:pos]

    if ( len(web) > 1 ) :
        cursor.execute('INSERT OR IGNORE INTO Webs (url) VALUES ( ? )', ( web, ) )
        cursor.execute('INSERT OR IGNORE INTO Pages (url, html, new_rank) VALUES ( ?, NULL, 1.0 )', ( starturl, ) )
        conn.commit()
else :
    print("Restarting existing crawl.  Remove spider.sqlite to start a fresh crawl.")


# Get the current webs
cursor.execute('''SELECT url FROM Webs''')
webs = list()
for row in cursor:
    webs.append(str(row[0]))

print(webs)

many = 0
while True:
    if ( many < 1 ) :
        sval = input('How many pages:')
        if ( len(sval) < 1 ) : break
        many = int(sval)
    many = many - 1

    try:
        row = selectRandomPage(cursor)
        # print row
        fromid = row[0]
        url = row[1]
    except:
        print('No unretrieved HTML pages found')
        many = 0
        break

    print('from_id:', fromid, url, end=' ')

    # If we are retrieving this page, there should be no links from it
    cursor.execute('DELETE from Links WHERE from_id=?', (fromid, ) )
    try:
        response = urlopen(url, context=ctx)

        responsebody = response.read()
        if response.getcode() != 200 :
            print("Error on page: ",response.getcode())
            cursor.execute('UPDATE Pages SET error=? WHERE url=?', (response.getcode(), url) )

        if 'text/html' != response.info().get_content_type() :
            print("Ignore non text/html page")
            cursor.execute('DELETE FROM Pages WHERE url=?', ( url, ) )
            conn.commit()
            continue

        print('(response characters: '+str(len(responsebody))+')', end=' ')

        html = BeautifulSoup(responsebody, "html.parser")
    except KeyboardInterrupt:
        print('')
        print('Program interrupted by user...')
        break
    except:
        print("Unable to retrieve or parse page")
        cursor.execute('UPDATE Pages SET error=-1 WHERE url=?', (url, ) )
        conn.commit()
        continue
    
    cursor.execute('INSERT OR IGNORE INTO Pages (url, html, new_rank) VALUES ( ?, NULL, 1.0 )', ( url, ) )
    cursor.execute('UPDATE Pages SET html=? WHERE url=?', (memoryview(responsebody), url ) )
    conn.commit()

    # Retrieve all of the anchor tags
    anchors = html('a')
    count = 0
    for anchor in anchors:
        href = anchor.get('href', None)
        if ( href is None ) : continue
        
        up = urlparse(href)

        # drop all anchors that are in page jumps
        if len(up.fragment) > 0 and up.path == '': continue

        # Resolve relative references like href="/contact"
        if ( len(up.scheme) < 1 ) :
            href = urljoin(url, href)

        ipos = href.find('#')
        if ( ipos > 1 ) : href = href[:ipos]

        if ( href.endswith('.png') or href.endswith('.jpg') or href.endswith('.gif') ) : continue
        if ( href.endswith('/') ) : href = href[:-1]
        # print href
        if ( len(href) < 1 ) : continue

		# Check if the URL is in any of the webs
        found = False
        for web in webs:
            if ( href.startswith(web) ) :
                found = True
                break
        if not found : continue

        cursor.execute('INSERT OR IGNORE INTO Pages (url, html, new_rank) VALUES ( ?, NULL, 1.0 )', ( href, ) )
        count = count + 1
        conn.commit()

        cursor.execute('SELECT id FROM Pages WHERE url=? LIMIT 1', ( href, ))
        try:
            row = cursor.fetchone()
            toid = row[0]
        except:
            print('Could not retrieve id')
            continue
        # print fromid, toid
        cursor.execute('INSERT OR IGNORE INTO Links (from_id, to_id) VALUES ( ?, ? )', ( fromid, toid ) )


    print('(Added ' + str(count) + ' outbound links)')

cursor.close()
