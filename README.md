# News-category-classifier
A Bidirectional LSTM and Attention model to classify news article.

Dataset can be downloaded from the website: https://www.kaggle.com/rmisra/news-category-dataset#News_Category_Dataset_v2.json

Refer to model.png for the model used.

Preprocessing steps:
1. Removal of punctuations.
2. Tokenization.

"json_embedding_news.json" file is created to store dictionary of GloVe embeddings for words in headline and short_description.

"dataset_final.txt" file is the final input to the model, each line in it is of the form: "headline" + "\t" + "short_description" + "category_index" + "\n"

"category_index.json" file contains dictionary to convert category to index vice versa.

## Authors

*  [Vishwajeet Kumar](https://github.com/vishwajeetkr)
*  [Atul Kumar](https://github.com/atkatul)
*  [Akash Meshram](https://github.com/akashmeshram)
