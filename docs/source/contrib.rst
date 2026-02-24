Contributing to the Package
===========================

Reporting Issues
----------------
If you uncover a bug or issue with the code, please open an issue through the GitHub site: https://github.com/hdrake/xwmb

Developing New Routines
-----------------------
Pull requests for new routines and code are welcome.

Creating a development environment
----------------------------------

.. code-block:: bash

    conda env update -f docs/environment.yml
    conda activate docs_env_xwmb
    pip install -e .

Locally building the documentation
----------------------------------

.. code-block:: bash

    conda activate docs_env_xwmb
    rm -rf docs/_build
    sphinx-build -W -b html docs/source docs/_build/html