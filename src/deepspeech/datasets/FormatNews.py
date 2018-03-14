# -*- coding:utf-8 -*-
from __future__ import print_function
import glob
from SegNews import doSeg
import re
import os
from optparse import OptionParser 
import traceback

if __name__ == '__main__':
    usage = "%prog --rootPath <input text file path>"
    parser = OptionParser(usage)
    parser.add_option('--rootPath', dest='rootPath', default='news_dataset', help='dataset root path')

    (o, args) = parser.parse_args()
    
    output_dir = o.rootPath + '/output'
    os.system('rm -rf %s' % output_dir)
    os.mkdir(output_dir)
    trs_files = glob.glob(o.rootPath + '/news-Annotation/*.trs')

    utter_dict = dict()

    for trs_file in trs_files:
        identity_part = trs_file.split('/')
        identity_part = identity_part[-1].split('.')
        try:
            another_pos = re.search('%s(.*?)%s' % ('\)\(', '\)'), identity_part[0]).group(1)
            identity_part = another_pos
        except:
            identity_part = identity_part[0]
        wav_file = glob.glob(o.rootPath + '/news-Audio/*' + identity_part + '*.wav')
        print('start file: %s' %(trs_file))
        if len(wav_file) == 1:
            wav_file = wav_file[0]
        else:
            print('wavfile not found')
            continue
        try:
            utter_dict = doSeg(trs_file, wav_file, output_dir, utter_dict, 'utf8')
            continue
        except:
            error_info = traceback.format_exc()
            try:
                utter_dict = doSeg(trs_file, wav_file, output_dir, utter_dict, 'gb2312')
            except:
                print('cannot decode infile: %s'%trs_file)
                print('error info\nutf8:')
                print(error_info)
                print('gb2312:')
                traceback.print_exc()
