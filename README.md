# DistilBERT-fine-tuning
In this project we fine-tune the DisilBERT model and compare it to a baseline SVM classifier on the multiclass task of classifying pro-eating disorder users on Twitter.

To set up project using the correct Python packages, run this command
```
pip install -r requirements.txt
```

`preprocess_tweets.ipynb` imports the original dataset of 9.821.107 tweets and aggregates them into 6824 rows of text document, where each row represents one Twitter-user.

`distilbert-finetune.py` includes the fine-tuning of the DistilBERT-base Sequence Classifier.

`DistilBERT_predictions.ipynb` includes importing the trained DistilBERT model and predictions.

`SVM_classifier.ipynb` includes the training and testing of the SVM classifier
