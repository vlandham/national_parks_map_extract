
jpgs:
	./tools/convert_pdf_to_jpg.sh

icons:
	python extract_icons.py

filter:
	python filter_bad_matches.py

key:
	python create_key_config.py

site:
	python prepare_for_site.py



