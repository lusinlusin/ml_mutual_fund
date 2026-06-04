# src/main.py
from datetime import datetime
import importlib
import time

try:
    from .config import TARGET_MODE
except ImportError:
    from config import TARGET_MODE


# Set any step to False to skip it.
RUN_DATA_DOWNLOAD = False
RUN_SAMPLE_FILTER = False
RUN_FEATURES = True
RUN_MODEL = True
RUN_BACKTEST = True

# The pipeline-wide prediction target (TARGET_MODE) is defined once in
# src/config.py and shared by features.py, model.py, and backtest.py.
# To switch between 'alpha' and 'excess_return', edit it there.


PIPELINE_STEPS = [
    ('data_download.py', RUN_DATA_DOWNLOAD, 'data_download'),
    ('sample_filter.py', RUN_SAMPLE_FILTER, 'sample_filter'),
    ('features.py', RUN_FEATURES, 'features'),
    ('model.py', RUN_MODEL, 'model'),
    ('backtest.py', RUN_BACKTEST, 'backtest'),
]


def import_local_module(module_name):
    if __package__:
        return importlib.import_module(f'.{module_name}', package=__package__)
    return importlib.import_module(module_name)


def run_step(label, module_name):
    start = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Running {label}")

    module = import_local_module(module_name)
    module.main()

    elapsed_minutes = (time.time() - start) / 60
    print(f"Finished {label} in {elapsed_minutes:.2f} minutes")


def main():
    pipeline_start = time.time()
    print(f"Pipeline target mode: {TARGET_MODE} (set in src/config.py)")

    for label, should_run, module_name in PIPELINE_STEPS:
        if not should_run:
            print(f"\nSkipping {label}")
            continue
        run_step(label, module_name)

    elapsed_minutes = (time.time() - pipeline_start) / 60
    print(f"\nPipeline finished in {elapsed_minutes:.2f} minutes")


if __name__ == '__main__':
    main()
