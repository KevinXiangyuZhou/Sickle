# Data Aggregator Synthesis


## Set up

Requirement: Python version >=3.7.

#### Create virtual environment (recommended)

[Virtual Environment](<https://docs.python.org/3/library/venv.html>) is recommended for managing dependencies (Conda doesn't work for now due to the dependency to z3 smt solver):

If using Virtual Environment:

   ```
   mkdir venv
   python3 -m venv ./venv
   source venv/bin/activate
   ```

At development time, use `source venv/bin/activate` (venv) or `source activate falx` (conda) to activate the virtual environment.

#### Install dependencies

1. Install python dependencies: `pip install -r requirements.txt`

2. Install Sickle in the development mode: `pip install -e .`

#### Benchmarks

See test_run function in file synthesizer_functionality_test.py to run individual test files.

#### Structure

operator nodes are in table_ast.py
synthesizer is constructed in synthesizer.py
Annotated table and util data structure are in table.py
