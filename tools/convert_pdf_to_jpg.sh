#!/bin/bash

indir='../data/npmaps'
infile='../data/npmaps/acadia-isle-au-haut-map.pdf'
outfile='../data/npmaps_jpg/acadia-isle-au-haut-map.jpg'

for f in $indir/*.pdf; do
  #echo $f
  filename=$(basename "$f")
  #echo $filename
  extension="${filename##*.}"
  bfilename="${filename%.*}"
  echo $bfilename
  output_filename="../data/npmaps_jpg/$bfilename.jpg"
  convert -density 300 -trim $f -quality 100 $output_filename
done

