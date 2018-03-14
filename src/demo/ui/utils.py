#/usr/bin/env python3
#-*- coding: utf-8 -*-


import pyaudio
import shutil
import wave
import numpy as np
import pylab as pl
import matplotlib as plt
from os import system, path
from queue import Queue, Empty
from socket import socket, timeout
from threading import Thread
from time import sleep
from threading import Lock


class Recognizer(object):
    """ Speech Recognition Interface """

    def __init__(self):
        self.status = 'off'

        self._speech_queue = Queue()
        self._result_queue = Queue()

        self._interval = 0.1  # sec
        self._recv_buffer_size = 4096

        # server
        self._root = 'audio/'
        self._remote_path = None

        # connection
        self._socket = None
        self._remote_addr = None
        self._remote_port = None


    def set_server(self, addr, port, user, path):
        self._remote_path = '%s@%s:%s' % (user, addr, path)
        self._remote_addr = addr
        self._remote_port = port

    def connect(self):
        assert self.status == 'off'
        assert self._socket == None
        self._socket = socket()
        self._socket.settimeout(self._interval)
        self._socket.connect((self._remote_addr, self._remote_port))

    def disconnect(self):
        assert self.status == 'off'
        assert self._socket is not None
        self._socket.close()
        self._socket = None

    def on(self):
        assert self.status == 'off'
        self.status = 'on'

        Thread(target=self._send_loop).start()
        Thread(target=self._recv_loop).start()

    def off(self):
        assert self.status == 'on'
        self.status = 'off'

    def put_speech(self, speech):
        assert self.status == 'on'
        self._speech_queue.put(speech)

    def get_result_nowait(self):
        assert self.status == 'on'
        return self._result_queue.get_nowait()

    def _scp_to_remote(self, filename):
        print('Exec: %s' % ('scp %s %s%s' % (filename, self._remote_path, filename)))
        system('scp %s %s' % (filename, self._remote_path))

    def _send_loop(self):
        while self.status == 'on':
            try:
                while True:
                    speech_filename = self._speech_queue.get_nowait()

                    self._scp_to_remote(self._root + speech_filename)

                    speech_filename = speech_filename.encode('utf-8')
                    self._socket.send(speech_filename+b'\n')
            except Empty:
                sleep(self._interval)


    def _recv_loop(self):
        self._clear_socket()
        while self.status == 'on':
            try:
                result = self._socket.recv(self._recv_buffer_size)
                result = result.decode('utf-8')
                result.strip('\n')
                self._result_queue.put(result)
            except timeout:
                pass  # socket timeout

    def _clear_socket(self):
        self._socket.settimeout(0.1)
        try:
            self._socket.recv(self._recv_buffer_size)
        except timeout:
            pass  # socket timeout
        self._socket.settimeout(self._interval)


class VoiceLoader(object):
    """ Load local file """
    # TODO
    def __init__(self):
        pass


class VoiceRecorder(object):
    """ Realtime Recorder """
    def __init__(self):
        self.status = 'off'

        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self._speech_queue = Queue()
        self._frame_queue = Queue()
        self._save_root = 'audio/'

        # voice format
        self._format = pyaudio.paInt16
        self._threshold = 500
        self._rate = 16000
        self._frame_size = 1024  # 1024 / 16000 = 0.064s
        self._channels = 1
        self._frame_length = float(self._frame_size) / float(self._rate)

        # speech
        self._min_sentence_length = 1  # sec
        self._min_sentence_frame_num = int(self._min_sentence_length / self._frame_length)
        self._min_pause_length = 0.5  # pause between sentences, sec
        self._min_pause_frame_num = int(self._min_pause_length / self._frame_length)
        # self._max_buffer_length = 2
        # self._max_buffer_frame_num = self._max_buffer_length / self._frame_length

        self._power_threshold = 0.0002
        self._zcr_threshold = 0.05
        self._auto_threshold_length = 2  # sec
        self._auto_threshold_frame_num = int(self._auto_threshold_length / self._frame_length)
        self._auto_threshold_dropout = 0.5
        self._auto_threshold_power_mult = 3
        self._auto_threshold_zcr_mult = 3

        self._noise = []
        self._noise_frame_num = 10

        # self._mutex = Lock()  # not needed

    def play(self, frame):
        stream = self._pyaudio.open(format=self._format, \
            channels=self._channels, rate=self._rate, input=False, \
            output=True, frames_per_buffer=self._frame_size)

        if isinstance(frame, list):
            for f in frame:
                stream.write(f)
        else:
            stream.write(frame)

        stream.close()

    def save(self, frame, filename):
        path = self._save_root + filename
        with wave.open(path, 'wb') as fout:
            fout.setparams((self._channels, 2, self._rate, 0, 'NONE', 'not compressed'))
            fout.writeframes(frame)


    def on(self, frame_preprocess=True):
        assert self.status == 'off'

        # start audio stream
        self._stream = self._pyaudio.open(format=self._format, \
            channels=self._channels, rate=self._rate, input=True, \
            output=False, frames_per_buffer=self._frame_size)

        # start recording
        self.status = 'on'

        Thread(target=self._record).start()

        if frame_preprocess:
            Thread(target=self._frame_preprocess).start()


    def off(self):
        assert self.status == 'on'

        self.status = 'off'
        self._stream.close()
        self._stream = None

        # clear queue
        try:
            while True:
                self._frame_queue.get_nowait()
        except Empty:
            pass

        try:
            while True:
                self._speech_queue.get_nowait()
        except Empty:
            pass

    def auto_set_threshold(self):
        assert self.status == 'off'

        print('auto setting threshold.')

        self.on(frame_preprocess=False)

        powers = []
        zcrs = []
        for i in range(self._auto_threshold_frame_num):
            frame = self._frame_queue.get()
            power, zcr = self._frame_power_zcr(frame)
            powers.append(power)
            zcrs.append(zcr)

        self.off()

        powers.sort()
        zcrs.sort()

        dropout = self._auto_threshold_dropout
        dropout_st = int(len(powers)*dropout*0.5)
        dropout_ed = int(len(powers)*(1 - dropout*0.5))

        powers = powers[dropout_st:dropout_ed]
        zcrs = zcrs[dropout_st:dropout_ed]

        self._power_threshold = self._auto_threshold_power_mult * sum(powers) / len(powers)
        self._zcr_threshold = self._auto_threshold_zcr_mult * sum(zcrs) / len(zcrs)

        print('power threshold:', self._power_threshold)
        print('zcr threshold:', self._zcr_threshold)

    def get_speech_nowait(self):
        return self._speech_queue.get_nowait()

    def set_save_root(self, root):
        self._save_root = root

    def _record(self):
        while self.status == 'on':  # read only, thread safe
            assert self._stream is not None
            frame = self._stream.read(self._frame_size)
            self._frame_queue.put(frame)

    def _frame_preprocess(self):  # frame -> sentences
        speech_frames = []
        background_frames = []
        while self.status == 'on':
            try:
                while True:
                    frame = self._frame_queue.get_nowait()
                    is_speech = self._is_speech(frame)
                    if is_speech:
                        if len(speech_frames) == 0 or len(background_frames) == 0:
                            speech_frames.append(frame)
                            background_frames.clear()
                        elif len(speech_frames) > 0 and len(background_frames) > 0:
                            speech_frames.extend(background_frames)
                            speech_frames.append(frame)
                            background_frames.clear()
                        else:
                            assert False  # impossible

                    if not is_speech:
                        if len(self._noise) == self._noise_frame_num:
                            self._noise = self._noise[1:]
                        self._noise.append(frame)  # modeling background noise

                        if len(speech_frames) == 0:
                            pass  # Do nothing
                        elif len(speech_frames) > 0:
                            background_frames.append(frame)

                    if len(background_frames) > self._min_pause_frame_num:
                        if len(speech_frames) > self._min_sentence_frame_num:
                            sentence = self._concat_frames(speech_frames)
                            # denoise
                            if self._noise:
                               sentence = self._denoise(sentence)
                            self._speech_queue.put(sentence)
                        background_frames.clear()
                        speech_frames.clear()
            except Empty:
                sleep(self._frame_length)


    def _frame_power_zcr(self, frame):
        numdata = self._frame_to_nparray(frame)
        power = self._power(numdata)
        zcr = self._zcr(numdata)
        return power, zcr

    def _frame_to_nparray(self, frame):
        assert self._format == pyaudio.paInt16
        numdata = np.fromstring(frame, dtype=np.int16)
        numdata = numdata / 2**15  # max val of int16 = 2**15-1
        return numdata

    def _nparray_to_frame(self, numdata):
        numdata = numdata * 2**15
        numdata = numdata.astype(np.int16)
        frame = numdata.tobytes()
        return frame
        

    def _power(self, numdata):
        return np.mean(numdata**2)

    def _zcr(self, numdata):
        zc = numdata[1:] * numdata[:-1] < 0
        zcr = sum(zc) / len(zc)
        return zcr

    def _is_speech(self, frame):
        power, zcr = self._frame_power_zcr(frame)
        voiced_sound = power > self._power_threshold
        unvoiced_sound =  zcr > self._zcr_threshold
        return voiced_sound or unvoiced_sound

    def _concat_frames(self, frames):
        return b''.join(frames)


    def _denoise(self, speech):
        # Spectral Subtraction
        speech_val = self._frame_to_nparray(speech)
        noise_val = self._frame_to_nparray(b''.join(self._noise))

        speech_fft_mag = np.abs(np.fft.fft(speech_val))
        noise_fft_mag = np.abs(np.fft.fft(noise_val))

        speech_freq = np.linspace(0, self._rate, len(speech_val))
        noise_freq = np.linspace(0, self._rate, len(noise_val))

        noise_fft_interp = np.interp(speech_freq, noise_freq, noise_fft_mag)

        denoised_fft_mag = np.maximum(speech_fft_mag - noise_fft_interp, np.zeros(speech_fft_mag.shape))

        denoised_fft = np.fft.fft(speech_val) * denoised_fft_mag / speech_fft_mag

        denoised_val = np.real(np.fft.ifft(denoised_fft))

        denoised = self._nparray_to_frame(denoised_val)
        return denoised


class Controller(object):
    def __init__(self):
        self.recognizer = Recognizer()
        self.recorder = VoiceRecorder()
        self.loader = VoiceLoader()

        self.set_server = self.recognizer.set_server
        self._root = None

        self.timer = 0

        self._status = None  # None, 'online', 'offline'
        self._texts = []
        
        self._cnt = 0

        self._interval = 0.1

    def set_root(self, root):
        self._root = root

    def get_texts(self):
        return self._texts[:]

    def clear_texts(self):
        assert self._status == None  # or use a mutex
        self._texts = []

    def connect(self):
        self.recognizer.connect()

    def disconnect(self):
        self.recognizer.disconnect()

    def online(self):
        assert self._status == None
        self._status = 'online'
        self.recorder.on()
        self.recognizer.on()
        Thread(target=self._online_loop).start()

    def offline(self):
        assert self._status == None
        self._status = 'offline'
        self.recognizer.on()
        Thread(target=self._offline_loop).start()

    def stop(self):
        status = self._status
        self._status = None

        if status == 'online':
            self.recorder.off()
            self.recognizer.off()

        if status == 'offline':
            self.recognizer.off()

    def send_offline_file(self, filename):
        filepath = self._root + filename
        assert path.exists(filepath)
        shutil.copy(filepath, filename)
        self.recognizer.put_speech(filename)

    def _online_loop(self):
        while self._status == 'online':
            result = None
            speech = None
            try:
                result = self.recognizer.get_result_nowait()
            except Empty:
                pass

            try:
                speech = self.recorder.get_speech_nowait()
            except Empty:
                pass

            if result:
                self._texts.append(result)
            if speech:
                filename = 'data%04d.wav' % self._cnt
                self.recorder.save(speech, filename)
                self.recognizer.put_speech(filename)
                self._cnt += 1
            if not result and not speech:
                sleep(self._interval)


    def _offline_loop(self):
        while self._status == 'offline':
            try:
                result = self.recognizer.get_result_nowait()
                self._texts.append(result)
            except Empty:
                sleep(self._interval)
