import os
from IPython.nbformat.current import reads
import gc

def check_notebook_with_assertion(file_name):
    from ipnbdoctest import test_notebook
    with open(file_name, 'r') as f:
        nb = reads(f.read(), 'json')
    failures, errors = test_notebook(nb, generate_png_diffs=True)
    assert(failures == 0 and errors == 0)


def test_notebooks_in_current_dir():
    # We use __file__ as the script is usually run from different directory
    # use relpath so the test names appear nicer and location independent in output
    current_directory = os.path.relpath(os.path.dirname(__file__))
    for dirpath, __, files in os.walk(current_directory):
        for file_name in files:
            if file_name.endswith('.ipynb'):
                yield check_notebook_with_assertion, os.path.join(dirpath, file_name)
                gc.collect()

