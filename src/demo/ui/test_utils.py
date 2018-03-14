#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
from queue import Queue, Empty
from time import sleep
from utils import VoiceRecorder



r = VoiceRecorder()
r.auto_set_threshold()
sleep(1)

r.on()
i = 0
while True:
    speech = r._speech_queue.get()
    r.save(speech, '%04d.wav' % i)
    i += 1
