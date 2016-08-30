#!/usr/bin/env python



import json
import glob
import os
from collections import Counter, defaultdict
from shutil import copyfile

INPUT_DIR = './data/icon_key'
INPUT_JSON = './data/icon_key/key.json'
OUTPUT_DIR = './data/symbol_keys'

def write_json(data, filename):
    '''
    Output data dict to json file
    '''
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def read_json(filename):
    '''
    '''
    data = {}
    with open(filename, 'r') as infile:
        data = json.load(infile)
    return data

def to_key(string):
    return string.strip().lower().replace(' ', '-').replace('\n', '-').replace('/', '-').replace('(', '').replace(')','')

def main():

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    symbols = read_json(INPUT_JSON)
    print(symbols)
    for symbol in symbols:
        key = to_key(symbol['label'])
        src_file = os.path.join(INPUT_DIR, symbol['basename'])
        out_file = os.path.join(OUTPUT_DIR, key + ".png")
        copyfile(src_file, out_file)
main()

