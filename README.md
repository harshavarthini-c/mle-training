# Configuration

    pyproject.toml is used for the configuration. Do python packaging.
# Refactored the code

    The code in nonstandardcode.py is split into four different files and stored in separate folder named as src/housing/
        1. run_script.py
        2. ingest_data.py
        3. train.py
        4. score.py
    By running this run_script.py, all other files will run. all function calls, argsparse are called and implemented respectively.
# Test function for the refactored code

    To test the entire code and installation, test files are written and stored at test/functional_test/, test/unit_test/, test/ folders.
        1. functional_test
            functional_test.py
        2. unit_test
            test_ingest_data.py
            test_run_script.py
            test_score.py
            test_train.py
        3. test_installation.py
        4. test_unitinstallation.py
# Docstring

    Add docstring for each python file on src/housing/
    use sphinx for the documentation.


## To excute the script
python run_script.py