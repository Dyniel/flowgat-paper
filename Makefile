PYTHON ?= python
PER_SEED_DIR ?= results/per_seed
FIGURES_DIR ?= results/figures

.PHONY: verify figures test zenodo-pack level2-diagnostics level3-eval level4-train paper

verify:
	$(PYTHON) scripts/verify_release.py

figures:
	$(PYTHON) src/make_fig_clinical_headline.py \
		--per_seed_dir $(PER_SEED_DIR) \
		--out_dir $(FIGURES_DIR)

test:
	$(PYTHON) -m pytest

zenodo-pack:
	bash .pack_zenodo.sh

level2-diagnostics:
	bash jobs/SUBMIT_E7_FOLLOWUP.sh

level3-eval:
	sbatch jobs/02_eval_all.sh
	sbatch jobs/04_dump_predictions.sh

level4-train:
	bash jobs/SUBMIT_ALL_SR.sh

paper:
	cd paper && latexmk -pdf main.tex
