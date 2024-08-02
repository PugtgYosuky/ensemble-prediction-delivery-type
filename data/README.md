# Datasets

This directory should contain two sub-directories called *images* and *tabular*.
The *tabular* directory should contain the .csv dataset for hte tabular patients history.
The images should be previously divided into folds and them into the different data to train, validation and test, before running the code.
*images* should be divided in directories one for each type of ultrasound. Inside them, it should have the *train*, *validation* and *test* directories. Each one of the divisions should have have the images related with a vaginal delivery or a cesarean section, each one in a different directory.

## Example of a division for the images dataset

1. dataset_fold_1
   1. Femur
      1. train
         1. Cesarean Birth
         2. Vaginal Birth
      2. validation
      3. test
   2. Abdomen
   3. Head
2. dataset_fold_2
3. dataset_fold_3

Note: In this example of tree files, only the first directory was extended to see its sub-directories, but all the should have the same structure.