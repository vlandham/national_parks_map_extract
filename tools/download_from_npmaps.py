#!/usr/bin/env python

import urllib
import os
from bs4 import BeautifulSoup
import time

CUR_DIR =  os.path.dirname(os.path.realpath(__file__))
page_location = os.path.join(CUR_DIR, '..', 'data', 'npmaps_uploads_index.html')
soup = BeautifulSoup(open(page_location), 'html.parser')

print(soup.title)

pdf_links = []
for link in soup.find_all('a'):
    #print(link.get('href'))
    href = link.get('href')
    if '.pdf' in href:
        pdf_links.append(href)


print(len(pdf_links))

prefix = "http://npmaps.com/wp-content/uploads/"

for filename in pdf_links:
    url = prefix + filename
    storage = os.path.join(CUR_DIR, '..', 'data', 'npmaps', filename)
    print(url)
    urllib.urlretrieve (url, storage)

    time.sleep(2)

