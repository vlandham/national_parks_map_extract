import os
import json
import glob
import matplotlib.pyplot as plt
from operator import itemgetter
from skimage import data, io, filters
import scipy.ndimage as ndimage
from skimage import color, img_as_float

from skimage.measure import compare_ssim as ssim

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import warnings
warnings.filterwarnings("ignore")

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


# Much of this code comes from:
# http://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
class BBox(object):
    '''
    Represents bounding box in image.
    Can be used to compare bounding boxes
    and extract image regions using the bounding box.

    '''
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
        '''
        Return width of bounding box
        '''
        return self.x2 - self.x1

    def height(self):
        '''
        Return height of bounding box
        '''
        return self.y2 - self.y1

    def x_extent(self):
        '''
        Return furthest x value of bounding box
        '''
        return self.x1 + self.width()

    def y_extent(self):
        '''
        Return furthest y value of bounding box
        '''
        return self.y1 + self.height()

    def extract(self, img):
        '''
        Extract sub-image from img where bounding box is
        '''
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
    '''
    Convert skimage slices to BBoxes
    '''
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


def remove_dups(bboxes):
    '''
    Remove duplicate BBoxes from a list of BBoxes
    '''
    clean_set = []
    for bbox in bboxes:
        if bbox not in clean_set:
            clean_set.append(bbox)
    return clean_set

def find_squares(bboxes, size=40, padding=10):
    '''
    Return filtered BBoxes list to include
    only BBoxes that are roughly square of size `size
    '''
    min_size = size - padding
    max_size = size + padding
    return [bbox for bbox in bboxes if (bbox.width() > min_size) and (bbox.width() < max_size) and (bbox.height() > min_size) and (bbox.height() < max_size)]

def find_img_match(img, imgs):
    '''
    Return index of most similar image in imgs to input img
    Uses: http://scikit-image.org/docs/dev/auto_examples/plot_ssim.html
    Code inspiration: http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    '''
    ssims = []
    for bimg in imgs:
        # get images into same shape.
        shape = bimg.shape
        img_shaped = img[0:shape[0], 0:shape[1], :]
        shape = img_shaped.shape
        bimg_shaped = bimg[0:shape[0], 0:shape[1], :]

        score = ssim(img_shaped, bimg_shaped, multichannel=True)
        ssims.append(score)

    # find max score
    #max_ssim = max(ssims)
    (ind, max_ssim) = max(enumerate(ssims), key=itemgetter(1))

    out = {"index":ind, "score":max_ssim}

    #return (min_ssim, ind)
    return out


def save_images(images, suffix, output_dir):
    '''
    Save a list of images to an output directory.
    If suffix is provided, it is appended to the end of the filename.

    Images are saved as PNG
    '''

    names = []
    for ind, image in enumerate(images):
        filename = output_dir + "/" + "icon_" + str(ind) + "_" + suffix +  ".png"
        io.imsave(filename, image)
        names.append(filename)
    return names

def convert_to_bw(img):
    '''
    Read in image from file,
    convert to pure Black only image
    '''

    #img = io.imread(filename, as_grey=False)
    #print(img.shape)


    fimg = img_as_float(img)

    filter_black = np.sum(fimg, axis=2)
    fimg_bw = fimg.copy()
    fimg_bw[(filter_black > 1)] = 1

    return fimg_bw


def extract_key(output_dir):
    '''
    Extracts key icons and text from symbols image
    '''

    filename = "./data/map_symbols.jpg"
    img = io.imread(filename, as_grey=False)
    fimg_bw = convert_to_bw(img)

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
    icons = [b.extract(img) for b in bboxes_filter]

    # extract labels of icons
    label_width = 400
    label_height = 50
    tlabels = []
    for bbox in bboxes_filter:
        label_box = BBox(bbox.x2, bbox.y1, bbox.x2 + label_width, bbox.y1 + label_height)
        tlabels.append(label_box.extract(img))
    print(len(tlabels))

    save_images(icons, "icon", output_dir)
    save_images(tlabels, "label", output_dir)

def load_keys(key_dir):
    '''

    '''
    names = glob.glob(key_dir + "/icon_*_icon.png")
    imgs = []
    for path in names:
        img = io.imread(path)
        fname = path.split("/")[-1].split(".")[0]
        imgs.append({"name":fname, "path":path, "img":img_as_float(img)})
    return imgs


def save_match_image(icons, key_imgs, match_inds, output_dir):
    rows = len(icons)
    cols = 2
    fig = plt.figure()
    plot_index = 1
    for ind, icon in enumerate(icons):
        ax = fig.add_subplot(rows, cols, plot_index)
        plot_index += 1
        plt.imshow(icon, cmap = plt.cm.gray)
        plt.axis("off")

        ax = fig.add_subplot(rows, cols, plot_index)
        plot_index += 1

        match = key_imgs[match_inds[ind]]

        plt.imshow(match, cmap = plt.cm.gray)
        plt.axis("off")
    plt.savefig(output_dir + '/matches.png')

def save_loc_image(bboxes, img, output_dir):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    plt.axis("off")
    im = ax.imshow(img)
    for bbox in bboxes:
        p = patches.Rectangle((bbox.x1, bbox.y1), bbox.width(), bbox.height(), fc = 'none', ec = 'red')
        ax.add_patch(p)
    plt.savefig(output_dir + '/locations.png')


def extract_icons(filename, keys):
    print filename
    map_name = filename.split('/')[-1].split('.')[0]
    output_dir = './data/out/' + map_name
    if os.path.isdir(output_dir):
        print "skipping: " + map_name
        return
    print map_name
    # ---
    #
    # ---
    img = io.imread(filename, as_grey=False)

    fimg_bw = convert_to_bw(img)
    print(fimg_bw.shape)

    fimg_bw_one = fimg_bw[:,:,1]
    fimg_bw_one_invert = 1 - fimg_bw_one

    labels, numobjects = ndimage.label(fimg_bw_one_invert)
    slices = ndimage.find_objects(labels)

    bboxes = slice_to_bbox(slices)

    bboxes = remove_dups(bboxes)
    print(len(bboxes))
    bboxes_filter = find_squares(bboxes)
    print(len(bboxes_filter))


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    icons = [b.extract(fimg_bw) for b in bboxes_filter]

    # ---

    img_names = save_images(icons, 'icon', output_dir)

    img_prefixs = [name.split('/')[-1].split('.')[0] for name in img_names]

    key_imgs = [key['img'] for key in keys]


    matches = [find_img_match(img, key_imgs) for img in icons]
    matches_ind = [m['index'] for m in matches]
    match_scores = [m['score'] for m in matches]
    match_names = [keys[match_ind]['name'] for match_ind in matches_ind]
    output = []
    for ind, icon in enumerate(icons):
        bbox = bboxes_filter[ind]
        out = {"icon_name": img_prefixs[ind],
                "filename": img_names[ind],
                "match_score": match_scores[ind],
                "match_index": matches_ind[ind],
                "match_name": match_names[ind],
                "position": {"x":bbox.x1, "y":bbox.y1, "width":bbox.width(), "height":bbox.height() }
              }
        output.append(out)
    out_filename = output_dir + "/info.json"

    root = {
            "map":map_name,
            "filename":filename,
            "icons":output,
            "img_size": {"width": fimg_bw.shape[1], "height": fimg_bw.shape[0]},
            }

    write_json(root, out_filename)

    save_match_image(icons, key_imgs, matches_ind, output_dir)
    save_loc_image(bboxes_filter, fimg_bw, output_dir)


key_dir = "./data/icon_key_good"
extract_key(key_dir)

keys = load_keys(key_dir)
print(len(keys))

#input_filename = './data/npmaps_jpg/death-valley-furnace-creek-map.jpg'
#extract_icons(input_filename, keys)

#input_filenames = glob.glob('./data/npmaps_jpg/*.jpg')
#for input_filename in input_filenames:
#    extract_icons(input_filename, keys)
