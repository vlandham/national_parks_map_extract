import os
import json
import glob
from operator import itemgetter
from skimage import data, io, filters
import scipy.ndimage as ndimage
from skimage import color, img_as_float

from skimage.measure import structural_similarity as ssim

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def x_extent(self):
        return self.x1 + self.width()

    def y_extent(self):
        return self.y1 + self.height()

    def extract(self, img):
        return img[(self.y1 - 1):self.y_extent(), (self.x1 - 1):self.x_extent(), :]


    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1

    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))

    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)


def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)


def remove_overlaps(bboxes):
    '''
    Return a set of BBoxes which contain the given BBoxes.
    When two BBoxes overlap, replace both with the minimal BBox that contains both.
    '''
    # list upper left and lower right corners of the Bboxes
    corners = []

    # list upper left corners of the Bboxes
    ulcorners = []

    # dict mapping corners to Bboxes.
    bbox_map = {}

    for bbox in bboxes:
        ul = (bbox.x1, bbox.y1)
        lr = (bbox.x2, bbox.y2)
        bbox_map[ul] = bbox
        bbox_map[lr] = bbox
        ulcorners.append(ul)
        corners.append(ul)
        corners.append(lr)

    # Use a KDTree so we can find corners that are nearby efficiently.
    tree = spatial.KDTree(corners)
    new_corners = []
    for corner in ulcorners:
        bbox = bbox_map[corner]
        # Find all points which are within a taxicab distance of corner
        indices = tree.query_ball_point(
            corner, bbox_map[corner].taxicab_diagonal(), p = 1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes.
                # Since we mutate the bbox, all references to this bbox in
                # bbox_map are updated simultaneously.
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1)
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2)
    return set(bbox_map.values())

def write_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def remove_dups(bboxes):
    clean_set = []
    for bbox in bboxes:
        if bbox not in clean_set:
            clean_set.append(bbox)
    return clean_set

def find_squares(bboxes):
    return [bbox for bbox in bboxes if (bbox.width() > 30) and (bbox.width() < 50) and (bbox.height() > 30) and (bbox.height() < 50)]

def find_img_match(img, imgs):
    #ssims = [ssim(img, bimg) for bimg in imgs]
    #ind = min(enumerate(ssims), key=itemgetter(1))[0]
    ind = 1
    return ind


def save_images(images, suffix, output_dir):

    names = []
    for ind, image in enumerate(images):
        filename = output_dir + "/" + "icon_" + str(ind) + "_" + suffix +  ".png"
        io.imsave(filename, image)
        names.append(filename)
    return names

def read_image(filename):

    img = io.imread(filename, as_grey=False)


    fimg = img_as_float(img)

    filter_black = np.sum(fimg, axis=2)
    fimg_bw = fimg.copy()
    fimg_bw[(filter_black > 1)] = 1

    return fimg_bw


def extract_key(output_dir):

    filename = "./data/map_symbols.jpg"
    fimg_bw = read_image(filename)

    fimg_bw_one = fimg_bw[:,:,1]
    fimg_bw_one_invert = 1 - fimg_bw_one

    labels, numobjects = ndimage.label(fimg_bw_one_invert)
    slices = ndimage.find_objects(labels)

    bboxes = slice_to_bbox(slices)

    bboxes = remove_dups(bboxes)
    print(len(bboxes))

    # limit to squares
    bboxes_filter = find_squares(bboxes)
    len(bboxes_filter)

    # extract icon images
    icons = [b.extract(fimg_bw) for b in bboxes_filter]

    # extract labels of icons
    label_width = 400
    label_height = 50
    tlabels = []
    for bbox in bboxes_filter:
        label_box = BBox(bbox.x2, bbox.y1, bbox.x2 + label_width, bbox.y1 + label_height)
        tlabels.append(label_box.extract(fimg_bw))
    print(len(tlabels))

    save_images(icons, "icon", output_dir)
    save_images(tlabels, "label", output_dir)

def load_keys(key_dir):
    names = glob.glob(key_dir + "/icon_*_icon.png")
    imgs = []
    for path in names:
        img = io.imread(path)
        fname = path.split("/")[-1].split(".")[0]
        imgs.append({"name":fname, "path":path, "img":img_as_float(img)})
    return imgs



def extract_icons(filename, keys):
    print filename
    map_name = filename.split('/')[-1].split('.')[0]
    print map_name
    fimg_bw = read_image(filename)

    fimg_bw_one = fimg_bw[:,:,1]
    fimg_bw_one_invert = 1 - fimg_bw_one

    labels, numobjects = ndimage.label(fimg_bw_one_invert)
    slices = ndimage.find_objects(labels)

    bboxes = slice_to_bbox(slices)

    bboxes = remove_dups(bboxes)
    print(len(bboxes))
    bboxes_filter = find_squares(bboxes)
    print(len(bboxes_filter))

    output_dir = './data/out/' + map_name

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    icons = [b.extract(fimg_bw) for b in bboxes_filter]

    img_names = save_images(icons, 'icon', output_dir)

    img_prefixs = [name.split('/')[-1].split('.')[0] for name in img_names]

    key_imgs = [key['img'] for key in keys]

    matches_ind = [find_img_match(img, key_imgs) for img in icons]
    match_names = [keys[match_ind]['name'] for match_ind in matches_ind]
    output = []
    for ind, icon in enumerate(icons):
        out = {"icon_name": img_prefixs[ind],
                "filename": img_names[ind],
                "match_index": matches_ind[ind],
                "match_name": match_names[ind],
                "position": {"x":bboxes[ind].x1 + (bboxes[ind].width() / 2), "y":bboxes[ind].y1 + (bboxes[ind].height() / 2)}

              }
        output.append(out)
    out_filename = output_dir + "/info.json"

    root = {
            "map":map_name,
            "filename":filename,
            "icons":output
            }

    write_json(root, out_filename)



key_dir = "./data/icon_key"
#extract_key(key_dir)

keys = load_keys(key_dir)
print(len(keys))

input_filename = './data/npmaps_jpg/death-valley-furnace-creek-map.jpg'
extract_icons(input_filename, keys)
