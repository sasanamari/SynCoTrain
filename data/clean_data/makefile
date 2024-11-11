
all: small_synthDF stabilityDF015

###############################################################################

synthDF:
	@read -p "Please specify your MPID: " MPID; \
		python makefile_synth.py --MPID $$MPID

small_synthDF: synthDF
	python makefile_small.py

stabilityDF015: synthDF
	python makefile_stability.py

###############################################################################

.SECONDARY:
