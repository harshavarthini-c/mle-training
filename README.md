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



## how to setup an environ

python3 -m pip install --user -U virtualenv
source my_env/bin/activate (To activate the environment)

## how to run files notebook/scripts

In virtualenv, we need to register it to Jupyter and give it a name
python3 -m ipykernel install --user --name=python3

jupyter notebook (To open a jupyter Notebook)
It gives us the server by opening your web browser to http://localhost:8888/

## Which model used and how did the modelling works.

We Linear Regression Model:
A Linear Regression model is trained using the LinearRegression class from Scikit-Learn.
1. The model is fitted to the prepared training data (housing_prepared) and corresponding labels (housing_labels).

2. Model Evaluation:
The trained Linear Regression model is evaluated on a subset of the training set (some_data) to see how well it performs.
Mean Squared Error (MSE) is calculated to measure the performance of the model on the entire training set.

3. Decision Tree Regressor:
A Decision Tree Regressor is trained using the DecisionTreeRegressor class from Scikit-Learn.
The model is fitted to the prepared training data (housing_prepared) and corresponding labels (housing_labels).
The model is then evaluated on the training set, and MSE is calculated.

4. Cross-Validation:
K-fold cross-validation is used to evaluate the Decision Tree model's performance more reliably.
Scikit-Learn's cross_val_score function is employed with the negative mean squared error as the scoring metric.

5. Random Forest Regressor:
A Random Forest Regressor is trained using the RandomForestRegressor class from Scikit-Learn.
The model is fitted to the prepared training data (housing_prepared) and corresponding labels (housing_labels).
The model is evaluated using cross-validation, and mean squared error is calculated.

6. Model Comparison:
The performances of the Linear Regression, Decision Tree, and Random Forest models are compared using cross-validation scores.

7. Model Fine-Tuning:
Grid Search is introduced to fine-tune hyperparameters for the Random Forest Regressor.
Different combinations of hyperparameter values are explored to find the best-performing model.

8. Analyzing Best Models:
Feature importance scores are extracted from the best-performing Random Forest model.
The importance of each feature is displayed, allowing for insights into the model's decision-making process.

9. Model Evaluation on Test Set:
The final model (best-performing Random Forest) is evaluated on a separate test set to estimate its generalization performance.
Mean Squared Error and Root Mean Squared Error are calculated.

10. Deployment and Monitoring:
The final model is deployed to a production environment.
A monitoring system is recommended to regularly check the model's live performance and trigger alerts if needed.
Automation is suggested for tasks such as collecting fresh data, retraining the model, and evaluating model performance.