#/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import time
import json
import random
from threading import Thread
from flask import Flask, render_template
from .utils import Controller


app_name = 'Speech Recognition'
app = Flask(app_name, template_folder='ui/templates')

remote_addr = ''
remote_port = 5141
remote_user = ''
remote_path = '~/chinese_speech_recognition/src/deepspeech/tmp/'

local_file_root = 'audio/'

controller = Controller()
controller.set_server(remote_addr, remote_port, remote_user, remote_path)
controller.set_root(local_file_root)
controller.connect()
controller.recorder.auto_set_threshold()
controller.recorder.set_save_root(local_file_root)

@app.route('/')
def index():
    global controller
    controller.stop()

    global remote_addr
    global remote_port
    remote_info = 'Connect with %s:%d' % (remote_addr, remote_port)

    page = 'index.html'
    return render_template(page, model_info=remote_info)


# Online
@app.route('/online')
def online():
    global controller
    controller.clear_texts()

    page = 'online.html'
    return render_template(page)

@app.route('/online/startstop', methods=['POST', 'GET'])
def online_start_stop():
    global controller

    if time.time() - controller.timer > 0.5:  # 0.5 sec
        controller.timer = time.time()

        if controller._status is None:
            controller.online()
        elif controller._status == 'online':
            controller.stop()
        else:
            assert False

    if controller._status is None:
        return 'off'
    if controller._status == 'online':
        return 'on'

@app.route('/online/text', methods=['GET'])
def online_text():
    global controller
    texts = controller.get_texts()
    return json.dumps([' '.join(texts)])

# Offline
@app.route('/offline')
def offline():
    page = 'offline.html'
    global controller
    controller.clear_texts()

    controller.offline()
    return render_template(page)

@app.route('/offline/open/<string:filename>', methods=['POST', 'GET'])
def offline_open(filename=None):
    assert filename is not None
    global controller

    controller.send_offline_file(filename)
    return filename

@app.route('/offline/text', methods=['GET'])
def offline_text():
    global controller
    texts = controller.get_texts()
    return json.dumps(texts)
