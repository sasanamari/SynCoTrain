
all: oqmd_df.pkl

###############################################################################

oqmd_raw_oxygen.npy:
	@echo "Jarvis is required to download OQMD data:"
	@echo "> conda create -n jarvis python=3.6 pip"
	@echo "> conda activate jarvis"
	@echo "> pip install jarvis"
	python makefile_oqmd.py

oqmd_df.pkl: oqmd_raw_oxygen.npy
	python makefile_pd.py

###############################################################################

.SECONDARY:
