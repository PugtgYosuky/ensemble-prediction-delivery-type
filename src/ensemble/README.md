# Ensemble of the image and tabular data predictions

## Files

- *merge-predictions.ipynb* : To combine the predictions, for all the folds, between a tabular and a image classifier, into a single file (one for each folder), so that the next notebook can ensemble the predictions.
- *ensemble-predictions.ipynb* : Makes the ensemble of the predictions between a tabular and a image classifier. It uses the output of the previous notebook. It creates a file for the results metrics and several images with the confusion matrices and ROC curves of the results.
  