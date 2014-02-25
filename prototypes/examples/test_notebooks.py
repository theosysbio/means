import os
from IPython.nbformat.current import reads

def check_notebook_with_assertion(file_name):
    from ipnbdoctest import test_notebook
    with open(file_name, 'r') as f:
        nb = reads(f.read(), 'json')
    failures, errors = test_notebook(nb, generate_png_diffs=False)
    assert(failures == 0 and errors == 0)


def test_notebooks_in_current_dir():
    for dirpath, __, files in os.walk("."):
        for file_name in files:
            if file_name.endswith('.ipynb'):

                yield check_notebook_with_assertion, os.path.join(dirpath, file_name)

