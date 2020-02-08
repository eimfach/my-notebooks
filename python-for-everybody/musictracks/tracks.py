import xml.etree.ElementTree as ET
import sqlite3

# one artist has many albums
# one album has many tracks
# one track has one genre

# a track has a title, length, rating, genre, album
# a genre has a name
# an album has a title and an artist
# an artist has a name

connection = sqlite3.connect('trackdb.sqlite')
cursor = connection.cursor()

# Make some fresh tables using executescript()
cursor.executescript('''
DROP TABLE IF EXISTS Artist;
DROP TABLE IF EXISTS Album;
DROP TABLE IF EXISTS Track;
DROP TABLE IF EXISTS Genre;

CREATE TABLE Artist (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE
);

CREATE TABLE Genre (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE
);

CREATE TABLE Album (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    artist_id  INTEGER,
    title   TEXT UNIQUE
);

CREATE TABLE Track (
    id  INTEGER NOT NULL PRIMARY KEY 
        AUTOINCREMENT UNIQUE,
    title TEXT  UNIQUE,
    album_id  INTEGER,
    genre_id  INTEGER,
    len INTEGER, rating INTEGER, count INTEGER
);
''')

# <key>Track ID</key><integer>369</integer>
# <key>Name</key><string>Another One Bites The Dust</string>
# <key>Artist</key><string>Queen</string>


def lookup(d, key):
    found = False
    for child in d:
        if found:
            return child.text
        if child.tag == 'key' and child.text == key:
            found = True
    return None


xmlTree = ET.parse('Library.xml')
tracks = xmlTree.findall('dict/dict/dict')

for entry in tracks:
    if (lookup(entry, 'Track ID') is None):
        continue

    artist = lookup(entry, 'Artist')
    album = lookup(entry, 'Album')
    genre = lookup(entry, 'Genre')

    trackName = lookup(entry, 'Name')
    count = lookup(entry, 'Play Count')
    rating = lookup(entry, 'Rating')
    length = lookup(entry, 'Total Time')

    if trackName is None or artist is None or album is None or genre is None:
        continue

    # print(trackName, artist, album, count, rating, length)

    # insert new artist and get its id
    cursor.execute('''INSERT OR IGNORE INTO Artist (name) 
        VALUES ( ? )''', (artist, ))
    cursor.execute('SELECT id FROM Artist WHERE name = ? ', (artist, ))
    artist_id = cursor.fetchone()[0]

    # insert new album and get its id
    cursor.execute('''INSERT OR IGNORE INTO Album (title, artist_id) 
        VALUES ( ?, ? )''', (album, artist_id))
    cursor.execute('SELECT id FROM Album WHERE title = ? ', (album, ))
    album_id = cursor.fetchone()[0]

    # insert new genre and get its id
    cursor.execute(
        "INSERT OR IGNORE INTO Genre (name) VALUES ( ? )", (genre, ))
    cursor.execute("SELECT id FROM Genre WHERE name = ? ", (genre, ))
    genre_id = cursor.fetchone()[0]

    cursor.execute('''INSERT OR REPLACE INTO Track
        (title, album_id, genre_id, len, rating, count) 
        VALUES ( ?, ?, ?, ?, ?, ? )''',
                   (trackName, album_id, genre_id, length, rating, count))

    connection.commit()

connection.close()
