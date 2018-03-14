# -*- coding:utf-8 -*-
from __future__ import print_function
from pydub import AudioSegment
import re
import io
import sys
import os

audiostart = u'audio_filename="' # this string identifies the audio file in Transcriber
audioend = u'" version="' # this string identifies the end of the audio file name
syncbeg = u'<Sync time="' # this string identifies the beginning of each time-alignment
section = u'<Section type="' # this string identifies the beginning of the file/section
turnbeg = u'<Turn' # this string identifies the beginning of a speaker turn
speakstart = u' speaker="' # this string identifies the speaker
starttime = u' startTime="'  # this string identifies the beginning of a timecoded speaker's turn
startend = u'" endTime="'  # this string identifies the end of a timecoded speaker's turn
speakend = u'" startTime' # this string identifies the beginning of a timecoded speaker's turn in a slightly different format
syncend = u'"/>' # this string identifies the end of a sync point
endtime = u'endTime="' # this string identifies the end of a timecoded speaker's turn in a slightly different format
endsep = u'" '  # this string identifies a separated end
endings = u'"' # this string identifies the end of a timecode


def processChsTextFile(txt):
    deleted_patterns = [u'\([^\(\)]*\)', u'\{[^\{\}]*\}', u'\[[^\[\]]*\]', u'（[^（）]*）',
                        u'【[^【】]*】', u'&lt;', u'&gt;']
    txt = txt.replace('\n', '')
    for pattern in deleted_patterns:
        txt = re.sub(pattern, '', txt)
    txt = u''.join(x for x in txt if (0x4E00 <= ord(x) <= 0x9FBB) or (0x3400 <= ord(x) <= 0x4DB5)
                  or (0x20000 <= ord(x) <= 0x2A6D6) or (0xF900 <= ord(x) <= 0xFA2D) or (0xFA30 <= ord(x) <= 0xFA6A)
                  or (0xFA70 <= ord(x) <= 0xFAD9) or (0x2F800 <= ord(x) <= 0x2FA1D))
    return txt

def doSeg(infile, wavfile, output_dir, utter_dict, encod):
    trsfile = io.open(infile, 'rt', encoding=encod)
    timecodes = []  # create a list value to keep track of timecodes
    speaker = []  # create a list value to keep track of speakers
    speakturn = u''  # create a string value to keep track of speaker turns
    speakvalstart = []  # create a list value to keep track of speaker start times
    speakvalend = []  # create a list value to keep track of speaker end times
    lines = []  # create a list value to keep track of text lines

    last_line_count = -1
    count = 0
    for line in trsfile:
        count += 1
        try:
            if audiostart in line:  # get the filename for the audio
                result = re.search('%s(.*?)%s' % (audiostart, audioend), line).group(1)
                audioname = result.replace(' ', '_')
        except:
            pass
        try:
            if section in line:  # get the endtime of the sound file
                complete = re.search('%s(.*?)%s' % (endtime, endings), line).group(1)  # story it in the 'complete' variable
        except:
            pass
        try:
            if turnbeg + starttime in line:  # get the first speaker's turn
                spone = re.search('%s(.*?)%s' % (starttime, endings + ' '), line).group(1)  # get the start of the speaker's turn
                sptwo = re.search('%s(.*?)%s' % (endtime, endings + ' '), line).group(1)  # get the end of the speaker's turn
                speakvalstart.append(spone)  # store it in a list
                speakvalend.append(sptwo)  # store it in another list
                speak = re.search('%s(.*?)%s' % (speakstart, endings), line).group(1)  # get the name of the speaker
                speakturn = speak  # set the current value of the string variable 'speakturn' to the speaker name
        except:
            pass
        try:
            if turnbeg + speakstart in line:  # get a non-first speaker's turn
                speak = re.search('%s(.*?)%s' % (speakstart, endings + ' '), line).group(
                    1)  # get the name of the speaker
                speakturn = speak  # set the current value of the string variable 'speakturn' to the speaker name
                spone = re.search('%s(.*?)%s' % (starttime, endings + ' '), line).group(
                    1)  # get the start of the speaker's turn
                sptwo = re.search('%s(.*?)%s' % (endtime, endings), line).group(1)  # get the end of the speaker's turn
                speakvalstart.append(spone)  # store it in a list
                speakvalend.append(sptwo)  # store it in another list
        except:
            pass
        try:
            if '<Sync' in line:  # if there is a sync point
                speaker.append(speakturn)  # add the name of the speaker to the 'speaker' list - this ensures that every turn has a corresponding speaker name
        except:
            pass
        try:
            if syncbeg in line:  # if there is a sync point
                sync = re.search('%s(.*?)%s' % (syncbeg, endings), line).group(1)  # get the timecode
                timecodes.append(sync)  # store it in the 'timecodes' list
        except:
            pass
        try: 
            if '<Who' in line:
                if count == last_line_count + 1:
                    last_line_count = count
        except:
            pass
        try:
            if '<' not in line:  # if there is non-html tagged text, this is the text associated with a turn
                this_line = processChsTextFile(line)
                if count == last_line_count + 1:
                    lines[-1] = lines[-1] + this_line
                else:
                    lines.append(this_line)  # add it to the 'lines' list
                last_line_count = count
        except:
            pass

    sound = AudioSegment.from_file(wavfile, format="wav")

    # begin output
    total_lines = len(lines)
    tmp_count = 0
    if not (len(lines) == len(speaker) and len(speaker) == len(timecodes)):
        print('warning: in this file, length of lines and speakers not match')
        return utter_dict

    for i in range(total_lines):
        if len(lines[i]) > 50 or len(lines[i]) < 5:
            continue
        tmp_count += 1
        if not utter_dict.has_key(lines[i]):
            utter_dict[lines[i]] = len(utter_dict.keys()) + 2001
        utter_id = utter_dict[lines[i]]
        speaker_id = speaker[i].replace(' ', '')
        output_str = speaker_id + '_' + str(utter_id) + '_' + str(tmp_count)
        start_pin = int(float(timecodes[i]) * 1000)
        if i != total_lines - 1:
            end_pin = int(float(timecodes[i+1]) * 1000)
        else:
            end_pin = int(float(complete) * 1000)
        frame = sound[start_pin: end_pin]
        frame.export(output_dir + '/' + output_str + '.wav', format="wav")
        output_file = io.open(output_dir + '/' + output_str + '.txt', 'wt', encoding='utf-8')
        output_file.write(lines[i])
        output_file.close()

    trsfile.close()  # close the trsfile now that all the data has been written to the 'textlines' lists/database
    print('finish this')
    return utter_dict
