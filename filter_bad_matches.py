
import os
import json
import glob

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

def main(input_dir, min_score):

    info_files = glob.glob(input_dir + "/**/info.json")
    for info_file in info_files:
        print info_file
        output_filename =  os.path.splitext(info_file)[0] + "_filter.json"
        print output_filename
        info = read_json(info_file)
        org_count = len(info['icons'])
        info['icons'] = [i for i in info['icons'] if i['match_score'] > min_score]
        print(info['map'])
        print( "{0} > {1}".format(str(org_count), str(len(info['icons']))))
        print('\n')
        write_json(info, output_filename)


main('./data/out', 0.3)





