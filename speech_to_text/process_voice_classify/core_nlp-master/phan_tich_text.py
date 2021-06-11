# from tokenization.crf_tokenizer import CrfTokenizer
#
# crf_tokenizer_obj = CrfTokenizer()
# # crf_tokenizer_obj.train('data/tokenized/samples/training')
# # Note: If you trained your model, please set correct model path and do not train again!
# crf_tokenizer_obj = CrfTokenizer(model_path='models/pretrained_tokenizer.crfsuite')
# # crf_tokenizer_obj = CrfTokenizer(model_path='models/testtrained_tokenizer.crfsuite')
# test_sent = "Dự thảo tập trung xây dựng"
# tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
# print(tokenized_sent)


# input_dir = 'data/word_embedding/samples/html'
# output_dir = 'data/word_embedding/real/training'
# from tokenization.crf_tokenizer import CrfTokenizer
# from word_embedding.utils import clean_files_from_dir
# crf_config_root_path = "./"
# crf_model_path = "models/testtrained_tokenizer.crfsuite"
# tokenizer = CrfTokenizer(config_root_path=crf_config_root_path, model_path=crf_model_path)
# clean_files_from_dir(input_dir, output_dir, should_tokenize=True, tokenizer=tokenizer)



from pyvi import ViTokenizer, ViPosTagger

m = ViTokenizer.tokenize(u"bật đèn tranh phòng khách")
print(m)



