# Prediction of the tabular data

## Files

- *main.py* : Main file to run the project, it uses the same initialization seed as the file to divide the image data into the different folds.
- *preprocess_data.py* : Auxiliary file to transform the input dataset. For example, contains the code to normalize the features, etc
- *run_models.py*: Auxiliary file to evaluate the different classifiers
- *utils.py*: Auxiliary file to instantiate the classifiers, and other tasks such as calculate final results (metrics).
- *config.json*: Configuration file to run the main script. It defines with dataset should be ran, which classifiers should be tested, how many folds, which variables should be normalized, etc. Section **Configuration File** describes its format and possible attributes.

## Run the tabular classifiers

    cd src/tabular-classifier
    python main.py config.json

Note: This will create a logs file where the results (metrics and predictions of each classifier will be stored)

## Configuration File

The configuration file contains the description of how the program should be run. It as the following attributes.

| **Attribute** | **Type Value** | **Description** | **Example** |
| ----- | ----- | ----- | ----- |
| dataset | String | Path to the tabular data. The given dataset should already be transformed in terms of categorical features and imputation of possible missing values | "../../data/tabular/{dataset_name}.csv" |
| target_column | String | Name of the target column | "Class" |
| models_names | List | List with the Models to evaluate and their hyperparameters. Each value of the list is a list where the first position is classifier's name and the second is a dict with the hyperparameters | [["AdaBoostClassifier" : {}], ["LogisticRegression", {}]] |
| kfolds | Integer | Number of folds to use to evaluate the models. Is should be equal to the ones used to divide the image dataset into folds | 3 |
| norm_model | String | Name of the normalization models used to transform the numerical data | "Robust" |
| numeric_features | List | List with the names of the numerical features to be normalized | ["var1", "var3"] |
