
## Installation

Create new conda environment.

```
conda create --python=2 mapextract
```

Check `requirements.txt` for python packages that need to be installed.
Install these packages

```
pip install -r requirements.txt
```

## Getting Data

Scripts expect map images to be in a certain location:

```
data
  |
  -- npmaps_jpg (JPG versions of PDF maps)
  |
  -- icon_key (all icons reference info derived from data/map_symbols.pdf)
  |
  -- out (output directory of scripts)
```

A few scripts are in place to assist with this process.

### Tools

```
tools/download_from_npmaps.py
```

Downloads all files found on [NPMaps](http://npmaps.com/) as of 07/2016.
Puts them in `data/npmaps`.

```
tools/convert_pdf_to_jpg.sh
```

Converts all pdf's found in `data/npmaps` to large JPG images and stores them in `data/npmaps_jpg`

I then manually filtered these jpgs to include only relevant maps.

## Running

Once `data/npmaps_jpg` is populated with full size jpg images of
