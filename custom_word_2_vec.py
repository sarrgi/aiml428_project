from gensim.models import Word2Vec

# define training data


# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = model.wv
print(len(words))
# access vector for one word
print(model.wv.get_vecattr('sentence', 'count'))
# save model
model.save('model.bin')


# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
