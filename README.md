# Naive_sentiment_analysis

- Here is the lab of CS438 in SJTU. We realize a simple classifier to judge if a comment to a goods is positive or not.

- Used model: 

  - CNN+LSTM for feature extraction
  - A trained word2vec model. (By Stanford GloVe project.)

- Some module need(may):

  ```
  Tensorflow1.12.0, jieba0.39, numpy1.15.2, gensim3.6.0, nltk3.4, scipy1.1.0, tqdm4.28.1
  ```

- To train the model:

  - Type ```python3 train_dl.py``` (if you want to change some parameters, go to ```config.py```)

- To evaluate and output a file:

  - Type ```python3 evaluate.py```(you may firstly refer our form.)