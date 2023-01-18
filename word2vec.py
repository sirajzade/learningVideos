import gensim
from gensim.models import Word2Vec

# import gensim.downloader as api
# print(api.info('text8'))
# model = Word2Vec(corpus)
# model.save("word2vec.model")

model1 = Word2Vec.load("word2vec.model")

# get the word which does not match in the group
print ("Find the one which does not match:")
words = "tea coffee car"
notmatching = model1.wv.doesnt_match(words.split(" "))
print (" \"" + notmatching + "\" is not matchning here!")

# get the most close 10 words in the vector space
word1 = "president"
print("10 most close words to \"" + word1 + "\":")
print(model1.wv.most_similar(word1))
word2 = "city"
print("10 most close words to \"" + word2 + "\":")
print(model1.wv.most_similar(word2))
word3 = "coffee"
print("10 most close words to \"" + word3 + "\":")
print(model1.wv.most_similar(word3))

##### Here we begin with our toy example ######
print("Here we begin with our toy example")
corpus = [
    ['friend', 'friend', 'friend', 'dog', 'friend', 'friend', 'friend'],
    ['friend', 'friend', 'friend', 'dog', 'friend', 'friend', 'friend'],
    ['friend', 'friend', 'friend', 'dog', 'friend', 'friend', 'friend'],
    ['friend', 'friend', 'friend', 'cat', 'friend', 'friend', 'friend'],
    ['friend', 'friend', 'friend', 'cat', 'friend', 'friend', 'friend'],
    ['friend', 'friend', 'friend', 'cat', 'friend', 'friend', 'friend']
]

model2 = Word2Vec(sentences=corpus, min_count=1, vector_size=3, window=2, epochs=10)

print ("Let us see the words in the model a.k.a. the vocabulary:")
for index, word in enumerate(model2.wv.index_to_key):
    if index == 100:
        break
    print(f"word #{index}/{len(model2.wv.index_to_key)} is {word} with vector {model2.wv.get_vector(word)}")

word4 = "cat"
print("most close words to \"" + word4 + "\":")
print(model2.wv.most_similar(word4))
word5 = "dog"
print("most close words to \"" + word5 + "\":")
print(model2.wv.most_similar(word5))

# This is the example on a sentece or document level
print("This is the example on a sentece or document level:")
document = "the president took the wrong decision"
for word in document.split(" "):
    print("10 most close words to \"" + word + "\":")
    similar = model1.wv.most_similar(word)
    for candidate in similar:
        print(candidate[0])


