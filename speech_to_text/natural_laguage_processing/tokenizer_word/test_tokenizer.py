import pickle

with open('tokenizer_S1.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# with open('drive/My Drive/Machine_Learning-prj/vietnamese_language_model/sequences_digit_s2_1M.pkl', 'rb') as f:
#     sequences_digit = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1





