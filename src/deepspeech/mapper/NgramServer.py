import kenlm
import socket
import os
from optparse import OptionParser 
import threading
import SocketServer
import math

HOST = "127.0.0.1"
PORT = 50963

usage = "%prog --languageModelPath <input text file path>"
parser = OptionParser(usage)
parser.add_option('--languageModelPath', dest='lmPath', default='datasets/thchs30/ngram.lm', help='language model path')

(o, args) = parser.parse_args()
if os.path.isfile('mapper/ngram.info.log'):
    exit(0)
if not os.path.isfile(o.lmPath):
    print("Error: %s does not exist" %(o.lmPath))
    exit(1)

languageModel = kenlm.Model(o.lmPath)
coeff = math.log(10)

class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024)
        if data == "pid" :
            message = str(os.getpid())
        else:
            score = languageModel.score(data, bos = False, eos = False)
            message = str(score * coeff)
        self.request.sendall(message)
        self.request.close()

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)

with open('mapper/ngram.info.log', 'w') as f:
    f.write(str(os.getpid()))

server.serve_forever()
