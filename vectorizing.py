import numpy as np
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

corpus = [
    'This is more about dogs and cats and bats.',
    'This document is discussing algorithms and data structures.',
    'dogs can be here, so the cats and the bats are also present.',
    'And you need algorithms in order to do data structures',
]
cvectorizer = CountVectorizer()
X = cvectorizer.fit_transform(corpus)
dictionary = cvectorizer.get_feature_names_out()

print("We will call all the tokens or words or also called terms of the files dictionary: ")
print(dictionary)
print("----------")
print("This is our Document Term Matrix: ")
print(X.toarray())
print("----------")
print("this is the data type of our matrix and it's shape. Shape shown the number of rows and columns: ")
print(type(X))
print(X.shape)

documentTermMatrix = pd.DataFrame(X.toarray(),
                                  index=["document 1", "document 2", "document 3", "document 4", ],
                                  columns=dictionary)

documentTermMatrixTransposed = pd.DataFrame(np.transpose(X.toarray()),
                                  index=dictionary,
                                  columns=["document 1", "document 2", "document 3", "document 4", ])

print("----------")
print("And here we have a beautiful layout for our Document Term Matrix with pandas: ")
print("----------")
print(documentTermMatrix.to_string())

print("----------")

print(documentTermMatrixTransposed.to_string())

print("----------")
# Here we run the SVD algorithm
svd = TruncatedSVD(n_components=2)
lsa = svd.fit_transform(X)
print("Some additional information after running SVD: ")
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)
print(svd.algorithm)
print("----------")
print("Here are svd components: ")
print(svd.components_)
print("----------")
print("Here is the resulting compressed matrix: ")
print(lsa)
print("----------")

topic_encoded_df: DataFrame = pd.DataFrame(lsa, columns=['topic_1', 'topic2'])

topic_encoded_df["corpus"] = corpus

print(topic_encoded_df.to_string())

encoding_matrix = pd.DataFrame(svd.components_,
                               index=['topic_1', 'topic_2'],
                               columns=dictionary).T
print("----------")
print("We reduced the number of documents: ")
print(documentTermMatrix.to_string())
print("----------")
print(encoding_matrix.T.to_string())
print("----------")
print("Let us see the most important words for the topics by sorting by their absolute values: ")
print("Here the words for the topic 1: ")
print(encoding_matrix.sort_values('topic_1', ascending=False))
print("----------")
print("Here the words for the topic 2: ")
print(encoding_matrix.sort_values('topic_2', ascending=False))
