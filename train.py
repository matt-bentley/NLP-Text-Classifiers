from data_helper import clean_str

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

import io

df_train = pandas.read_csv('data/data.csv', header=None, sep='\t', names=['source', 'title'])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = df_train['title']
trainDF['text'] = trainDF['text'].apply(clean_str)
trainDF['label'] = df_train['source']

# # load the dataset
# data = open('data/corpus').read()
# labels, texts = [], []
# for i, line in enumerate(data.split("\n")):
#     content = line.split()
#     labels.append(content[0])
#     texts.append(content[1])

# print(str(len(texts)) + ' lines of data')

# # create a dataframe using texts and lables
# trainDF = pandas.DataFrame()
# trainDF['text'] = texts
# trainDF['label'] = labels

# split the dataset into training and validation datasets 75/25
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.25, random_state=42)

print('Train count: ' + str(len(train_x)))
print('Evaluation count: ' + str(len(valid_x)))

print('')

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print('Processing Count vectors...')

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

print('Processing Tf-Idf vectors...')

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2))
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,7))
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# print('Creating word embedding matrix...')

# # load the pre-trained word-embedding vectors 
# embeddings_index = {}
# fin = io.open('data/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
# n, d = map(int, fin.readline().split())
# for i, line in enumerate(fin):
#     values = line.split()
#     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# # create a tokenizer 
# token = text.Tokenizer()
# token.fit_on_texts(trainDF['text'])
# word_index = token.word_index

# # convert text to sequence of tokens and pad them to ensure equal length vectors 
# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
# valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# # create token-embedding mapping
# embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


def train_model(classifier, feature_vector_train, labels, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, labels)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


print('')

print('Training Naive Bayes...')

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy)

print('')

print('Training Logistic Regression...')

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)

# print('')

# print('Training SVM...')

# # SVM on Count Vectors
# accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
# print("SVM, Count Vectors: ", accuracy)

# # SVM on Word Level TF IDF Vectors
# accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
# print("SVM, WordLevel TF-IDF: ", accuracy)

# # SVM on Ngram Level TF IDF Vectors
# accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# print("SVM, N-Gram Vectors: ", accuracy)

# # SVM on Character Level TF IDF Vectors
# accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
# print("SVM, CharLevel Vectors: ", accuracy)

print('')

print('Training SVM Linear...')

lin_svc = svm.LinearSVC(random_state=111, loss='hinge')
# SVM on Count Vectors
accuracy = train_model(lin_svc, xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors: ", accuracy)

# SVM on Word Level TF IDF Vectors
accuracy = train_model(lin_svc, xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, WordLevel TF-IDF: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(lin_svc, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)

# SVM on Character Level TF IDF Vectors
accuracy = train_model(lin_svc, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("SVM, CharLevel Vectors: ", accuracy)

print('')

print('Training SVM SGD...')

sgd = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-6, max_iter=25, random_state=42)
# SVM on Count Vectors
accuracy = train_model(sgd, xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors: ", accuracy)

# SVM on Word Level TF IDF Vectors
accuracy = train_model(sgd, xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, WordLevel TF-IDF: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(sgd, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)

# SVM on Character Level TF IDF Vectors
accuracy = train_model(sgd, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("SVM, CharLevel Vectors: ", accuracy)

print('')

print('Training Random Forest...')

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=50), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=50), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy)

# RF on Ngram Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=50), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, N-Gram Vectors: ", accuracy)

# RF on Character Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=50), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("RF, CharLevel Vectors: ", accuracy)

print('')

print('Training XGB...')

# Extereme Gradient Boosting on Count Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print("Xgb, Count Vectors: ", accuracy)

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("Xgb, WordLevel TF-IDF: ", accuracy)

# RF on Ngram Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Xgb, N-Gram Vectors: ", accuracy)

# # Extereme Gradient Boosting on Character Level TF IDF Vectors
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
# print("Xgb, CharLevel Vectors: ", accuracy)