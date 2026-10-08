PYTHON ?= python
TRIALS ?= 5
SMOKE_CASES ?= cases/happy_03_*.yaml,cases/policy_01_*.yaml,cases/toolfail_01_*.yaml

.PHONY: install check smoke eval report test

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) scripts/check_model.py

smoke:
	$(PYTHON) run_eval.py --trials 1 --cases "$(SMOKE_CASES)" --out results/smoke-latest --label smoke
	$(PYTHON) report.py --run-dir results/smoke-latest

eval:
	$(PYTHON) run_eval.py --trials $(TRIALS)

report:
	$(PYTHON) report.py

test:
	$(PYTHON) -m pytest -q
