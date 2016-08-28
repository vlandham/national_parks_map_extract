
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import glob
import os
import json


def write_json(data, filename):
    '''
    Output data dict to json file
    '''
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def main(input_dir):
    icon_filenames = glob.glob(input_dir + "/icon_*_icon.png")

    output_filename = input_dir + "/key.json"
    data = []

    for filename in icon_filenames:
        basename = filename.split("/")[-1]
        entry = {'filename': filename, 'basename':basename}
        label_name = filename.replace('_icon.png', '_label.png')
        #label_name =  os.path.splitext(filename)[0]
        label_text = pytesseract.image_to_string(Image.open(label_name))
        print(label_text)
        entry["label"] = label_text
        data.append(entry)

    write_json(data, output_filename)

main('./data/icon_key')

