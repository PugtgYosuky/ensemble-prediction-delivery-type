# Prediction of the image data

## Files

- *main.py* : Main script to run the image classifiers
- *utils.py* : Auxiliary file
- *config_{\*}.json*: Configuration files for the three types of images sources: Abdomen, Femur, Head

## Run the image classifiers

    cd src/image-classifier
    python main.py config_{*}.json

Note 1: For each fold and each type of ECO, the script should be run. E.g., with 3-folds and 3 types of ECOS (Abdomen, Femur, Head), 9 experiences should be run.

Note 2: This will create a logs file where the results (metrics and predictions of each classifier will be stored)

## Configuration file

The configuration file contains the description of how the program should be run. It as the following attributes.

| **Attribute** | **Type Value** | **Description** | **Example** |
| ----- | ----- | ----- | ----- |
| metrics | List | List with the metrics to use to train the DL model | ["accuracy"] | 
| optimizer | String | Name of the optimizer to train the DL model | "Adam" |
| learning_rate | Float | Learning rate to update model's weights | 0.001 |
| deep-model | String | Name of the DL model. Possible values: "inception", "resnet50", "vgg19", "xception" | "inception" | 
| loss | String | Loss function | "BinaryCrossentropy" |
| epochs | Integer | Number of epochs to train the models | 1500 |
| patience | Integer | Number of iterations without improving until stopping  | 200 |
| batch_size | Integer | Batch Size | 32 |
| width | Integer | Input width | 720 |
| height | Integer | Input height | 720 |
| resizing-width | Integer | Resize width | 80 |
| resizing-heigh | Integer | Resize height | 80 |
| transfer-learning | Boolean | To use transfer learning or not | True |
| weights | String | Weights to use to train the DL model | "imagenet" |
| dataset | String | Path to the dataset to use. The given folder should internally be divided into train, validation, test folder and each of them should contain the images for "Cesarean Birth" and "Vaginal Birth", respectively
