
all: wbm_oxides

###############################################################################

2022-10-19-wbm-computed-structure-entries+init-structs.json.bz2:
	wget -O $@ https://figshare.com/ndownloader/files/40344469

###############################################################################

wbm_oxides.pkl: 2022-10-19-wbm-computed-structure-entries+init-structs.json
	python makefile_wbm.py

###############################################################################

wbm_oxides: wbm_oxides.pkl
	python makefile_wbm_poscar.py

###############################################################################

%: %.bz2
	bunzip2 $<

###############################################################################

.SECONDARY:
