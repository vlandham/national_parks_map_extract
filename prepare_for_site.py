
import json
import glob
import os
from collections import Counter, defaultdict

INPUT_DIR = './data/out'
KEY_CONFIG = './data/icon_key/key.json'
PARK_INFO =  './data/national_parks.json'

OUTPUT_DIR = './data/parks'

SPECIAL_IDS = {'cuyahoga-valley': ['cuyahoga']}

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
    return string.strip().lower().replace(' ', '-').replace('\n', '-').replace('/', '-')

def get_parks_info():
    parks_info = read_json(PARK_INFO)
    for info in parks_info:
        info['name'] = info['Name']
        info['id'] = to_key(info['name'])
    return parks_info

def get_symbols():
    filenames = glob.glob(os.path.join(INPUT_DIR, '**', 'info_filter.json'))
    symbols = {}
    for filename in filenames:
        info = read_json(filename)
        symbols[info['map']] = info
    return symbols

def get_id_for(map_name, park_ids):
    for pid in park_ids:
        if pid in map_name:
            return pid
    if pid in SPECIAL_IDS:
        specials = SPECIAL_IDS[pid]
        for pid_section in specials:
            if pid_section in map_name:
                return pid
    return None

def group_by_park(parks_info, map_symbols):

    park_ids = [p['id'] for p in parks_info]

    for key, pmap in map_symbols.iteritems():
        pmap['park'] = get_id_for(pmap['map'], park_ids)
        if not pmap['park']:
            print('ERROR: no park for ' + pmap['map'])

    for info in parks_info:
        pid = info['id']
        maps = [map_symbol for map_symbol in map_symbols.itervalues() if map_symbol['park'] == pid]
        if not maps:
            print('NO MAPS FOR: ' + pid)
        info['maps'] = maps

    return parks_info

def get_key():
    key = read_json(KEY_CONFIG)
    key_dict = {}
    for symbol in key:
        k_id = symbol['basename'].split('.')[0]
        symbol['name'] = symbol['label']
        symbol['id'] = to_key(symbol['name'])
        key_dict[k_id] = symbol
    return key_dict

def counter_to_array(counter, name_map):
    counts = []
    for key, count in counter.iteritems():
        count = {'id':key, 'count':count, 'name':name_map[key]}
        counts.append(count)
    return sorted(counts, key=lambda x: x['count'], reverse=True)


def add_symbols(parks):

    keys = get_key()

    symbol_totals = Counter()
    symbol_parks = defaultdict(list)
    symbol_maps = defaultdict(list)


    park_symbols = {}
    for park in parks:
        park_totals = Counter()
        for pmap in park['maps']:
            pmap['symbols'] = []
            pmap['filename'] = pmap['filename'].split('/')[-1]
            map_totals = Counter()
            for icon in pmap['icons']:
                if icon['match_name'] in keys:
                    key = keys[icon['match_name']]
                    symbol = {}
                    symbol['pos'] = icon['position']
                    symbol['id'] = key['id']
                    symbol['name'] = key['name']
                    pmap['symbols'].append(symbol)
                    park_symbols[symbol['id']] = symbol['name']
                    map_totals[symbol['id']] += 1
                    park_totals[symbol['id']] += 1
                    symbol_totals[symbol['id']] += 1

                    #if park['id'] not in symbol_parks[symbol['id']]:
                    #    symbol_parks[symbol['id']].append(park['id'])
                    #if pmap['map'] not in symbol_maps[symbol['id']]:
                    #    symbol_maps[symbol['id']].append(pmap['map'])
                else:
                    print('WARNING: ' + icon['match_name'] + ' not valid symbol')

            for key, count in map_totals.iteritems():
                symbol_maps[key].append({'id':pmap['map'], 'count':count, 'name':park_symbols[key]})
            del(pmap['icons'])
            pmap['totals'] = counter_to_array(map_totals, park_symbols)

            # end map loop

        for key, count in park_totals.iteritems():
            symbol_parks[key].append({'id':park['id'], 'count':count})

        park['totals'] = counter_to_array(park_totals, park_symbols)
        park['map_count'] = len(park['maps'])
        park['symbol_count'] = sum(park_totals.values())
        # end park loop
    symbols = {}
    symbols['totals'] = counter_to_array(symbol_totals, park_symbols)
    symbols['parks'] = symbol_parks
    #symbols['maps'] = symbol_maps
    return {'symbols': symbols, 'parks':parks}


def output_parks(parks, output_dir):
    for park in parks:
        full_path = os.path.join(output_dir, park['id'])
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        park_filename = os.path.join(full_path, 'symbols.json')
        write_json(park, park_filename)


def get_park_ids(parks):
    ids = {}
    for park in parks:
        ids[park['id']] = {'name': park['name'],
                           'id': park['id'],
                           'map_count': park['map_count'],
                           'symbol_count': park['symbol_count']
                           }
    return ids


def main():
    parks_info = get_parks_info()
    print('parks #: ' + str(len(parks_info)))

    map_symbols = get_symbols()
    print('maps #: ' + str(len(map_symbols.keys())))

    parks = group_by_park(parks_info, map_symbols)

    parks = add_symbols(parks)

    output_filename = os.path.join(OUTPUT_DIR, 'all.json')
    write_json(parks, output_filename)

    output_filename = os.path.join(OUTPUT_DIR, 'all_parks.json')
    write_json(parks['parks'], output_filename)

    output_filename = os.path.join(OUTPUT_DIR, 'all_symbols.json')
    write_json(parks['symbols'], output_filename)

    park_ids = get_park_ids(parks['parks'])
    output_filename = os.path.join(OUTPUT_DIR, 'park_ids.json')
    write_json(park_ids, output_filename)


    output_parks(parks['parks'], OUTPUT_DIR)




main()
