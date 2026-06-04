# src/config.py
"""Shared pipeline configuration — single source of truth.

Settings that must stay consistent across features.py, model.py, and
backtest.py live here. Edit values in this file only; every script reads them,
whether the pipeline is run via main.py or a script is run standalone
(e.g. `python src/model.py`).
"""

# Supervised prediction target for the whole pipeline. Must match across the
# feature build, the model, and the backtest, so it is defined once here.
#   'alpha'         -> predict next-year annualized FF5+MOM realized alpha (baseline)
#   'excess_return' -> predict next-year annualized excess return (experiment)
TARGET_MODE = 'alpha'

if TARGET_MODE not in {'alpha', 'excess_return'}:
    raise ValueError("TARGET_MODE must be 'alpha' or 'excess_return'")
