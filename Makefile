.PHONY: setup verify patch clean-eval robustness-rtx4090 robustness-t4x2 tables figures test

setup:
	bash scripts/setup_environment.sh

verify:
	PYTHONPATH=src python scripts/verify_repository.py

verify-full:
	PYTHONPATH=src python scripts/verify_repository_full.py

patch:
	PYTHONPATH=src python scripts/patch_ultralytics.py

clean-eval:
	PYTHONPATH=src python scripts/evaluate_clean.py --data "$${DATA_YAML}"

robustness-rtx4090:
	bash scripts/reproduce_rtx4090.sh

robustness-t4x2:
	bash scripts/reproduce_kaggle_t4x2_target.sh

tables:
	python scripts/regenerate_all_tables.py

figures:
	python scripts/generate_figures.py

test:
	python scripts/verify_repository_full.py
