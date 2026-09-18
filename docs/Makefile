REPORT := xuannv_annual5m_report
TECTONIC ?= $(shell command -v tectonic 2>/dev/null)
ifeq ($(strip $(TECTONIC)),)
TECTONIC := $(HOME)/.local/bin/tectonic
endif

.PHONY: report
report:
	@if [ -x "$(TECTONIC)" ]; then \
		"$(TECTONIC)" --keep-logs --keep-intermediates "$(REPORT).tex"; \
	elif command -v xelatex >/dev/null 2>&1; then \
		xelatex -interaction=nonstopmode -halt-on-error "$(REPORT).tex" && \
		xelatex -interaction=nonstopmode -halt-on-error "$(REPORT).tex" && \
		xelatex -interaction=nonstopmode -halt-on-error "$(REPORT).tex"; \
	else \
		printf '%s\n' 'Install Tectonic or XeLaTeX with ctex, Fandol and TikZ; or set TECTONIC=/path/to/tectonic.'; \
		exit 1; \
	fi
