import http.client, subprocess

c = http.client.HTTPConnection('localhost', 8080)
c.request('POST', '/return', '{}')
doc = c.getresponse().read()
print (doc)