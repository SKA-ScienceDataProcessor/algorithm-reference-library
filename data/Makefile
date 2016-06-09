
DATA_FILE=data.tar.gz
DATA_URL=http://www.mrao.cam.ac.uk/~pw410/crocodile/$(DATA_FILE)

data_extracted : $(DATA_FILE)
	tar xzf $(DATA_FILE)
	touch data_extracted

$(DATA_FILE) : $(DATA_FILE).md5
	if [ "`md5sum $(DATA_FILE) 2>&1 || true`" != "`cat $(DATA_FILE).md5`" ]; then \
		wget $(DATA_URL) -O $(DATA_FILE); \
	fi
	touch $(DATA_FILE)
