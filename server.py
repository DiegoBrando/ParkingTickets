import subprocess
from bottle import run, post, request, response, get, route

@route('/<path>',method = 'POST')
def process(path):
    return subprocess.call(['python',path+'.py'],shell=True)

run(host='localhost', port=8080, debug=True)