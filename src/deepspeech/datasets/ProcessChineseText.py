#! /usr/bin/env python3
# -*- coding: utf8 -*-

import re
from optparse import OptionParser 

deleted_patterns = [r'\([^\(\)]*\)', r'\{[^\{\}]*\}', r'\[[^\[\]]*\]', r'（[^（）]*）',
                    r'【[^【】]*】']
line_symbols = ['\.', ',', '\?', '!', '。', '，', '？', '！', '…', ';', '；']

def processChsTextFile(in_filepath, out_filepath):
    with open(in_filepath, 'rt', encoding='utf-8') as infile, open(out_filepath, 'wt', encoding='utf-8') as outfile:
        txt = infile.read()
        txt = txt.replace('\n', '')
        txt = txt.replace(u'<contenttitle>', '(')
        txt = txt.replace(u'</contenttitle>', ')')
        for pattern in deleted_patterns:
            txt = re.sub(pattern, '', txt)
        txt = ''.join(x for x in txt if x in line_symbols or (0x4E00 <= ord(x) <= 0x9FBB) or (0x3400 <= ord(x) <= 0x4DB5)
                or (0x20000 <= ord(x) <= 0x2A6D6) or (0xF900 <= ord(x) <= 0xFA2D) or (0xFA30 <= ord(x) <= 0xFA6A)
                or (0xFA70 <= ord(x) <= 0xFAD9) or (0x2F800 <= ord(x) <= 0x2FA1D))
        txt = re.split('|'.join(line_symbols), txt)
        for line in txt:
            if len(line) >= 4:
                outfile.write(' '.join(line))
                outfile.write('\n')

if __name__ == "__main__":
    usage = "%prog --in-file <input text file path> --out-file <output text file path>"
    
    parser = OptionParser(usage)
    parser.add_option('--in-file', dest='in_file', help='input text file path')
    parser.add_option('--out-file', dest='out_file', help='output text file path')

    (o, args) = parser.parse_args()

    processChsTextFile(o.in_file, o.out_file)
    print("Done")
