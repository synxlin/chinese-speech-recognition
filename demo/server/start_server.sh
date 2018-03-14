#!/bin/bash
cd ~/chinese_speech_recognition/src/deepspeech
rm -rf tmp
mkdir tmp
th PredictServer.lua -modelPath exp/20170520_194309/model_epoch_30_20170520_194309_deepspeech.t7 -audioRoot tmp/ -cudnnFastest -beamSize 10 -lmAlpha 3
rm -rf tmp
rm -rf mapper/ngram.info.log
