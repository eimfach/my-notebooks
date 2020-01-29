import socket

# we basically simulate what is going to happen in a web browser

mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysocket.connect(("data.pr4e.org", 80))

# encode does turn the string from internal unicode to utf-8 bytes
cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\n\n'.encode()

mysocket.send(cmd)

while True:
    # retrieve blocks of 512 characters or bytes
    data = mysocket.recv(512)
    if (len(data) < 1):
        break
    print(data.decode())

mysocket.close()