from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.pipeline import EntityRuler
from spacy.training.example import Example
from colorama import Fore, Style, Back


TRAIN_DATA = [
    ('thành phố hồ chí minh chỉ đạo phòng chống dịch.', {'entities': [(0, 21, 'LOC')]}),
    ('Để đối phó với dịch Hải Phòng quyết tâm đến cùng.', {'entities': [(20, 29, 'LOC')]}),
    ('hôm nay thời tiết có mưa không', {'entities': [(0, 7, 'TIME'), (8, 17, 'WEATHER'), (18, 24, 'WEATHER')]}),
    ('thời tiết Hà Nội hôm nay thế nào.', {'entities': [(0, 9, 'WEATHER'), (10, 16, 'LOC'), (17, 24, 'TIME')]}),
    ('hôm nay Đà Nẵng có mưa không', {'entities': [(0, 7, 'TIME'), (8, 15, 'LOC'), (16, 22, 'WEATHER')]}),
    ('Có mưa ở Đà Nẵng không', {'entities': [(0, 6, 'WEATHER'), (9, 15, 'LOC')]}),
    ('Đà Nẵng quyết tâm đối phó với dịch bệnh', {'entities': [(0, 7, 'LOC')]})
]

model = None
output_dir = Path("vi_named_entity_recognition_model")
# n_iter = 200
#
# if model is not None:
#     nlp = spacy.load(model)
#     print("Loaded model '%s'" % model)
# else:
#     nlp = spacy.blank('en')
#     print("Created blank 'en' model")
#
# if 'ner' not in nlp.pipe_names:
#     # ner = nlp.create_pipe('ner')
#     ner = nlp.add_pipe('ner')     #, last=True)
# else:
#     ner = nlp.get_pipe('ner')
#
# for _, annotations in TRAIN_DATA:
#     for ent in annotations.get('entities'):
#         ner.add_label(ent[2])
#
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# with nlp.disable_pipes(*other_pipes):  # only train NER
#     optimizer = nlp.begin_training()
#     for itn in range(n_iter):
#         random.shuffle(TRAIN_DATA)
#         losses = {}
#         for text, annotations in tqdm(TRAIN_DATA):
#             doc = nlp.make_doc(text)
#             example = Example.from_dict(doc, annotations)
#             nlp.update(
#                 [example],
#                 # [annotations],
#                 drop=0.5,
#                 sgd=optimizer,
#                 losses=losses)
#         print(losses)
#
# for text, _ in TRAIN_DATA:
#     doc = nlp(text)
#     print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#     print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
#
# if output_dir is not None:
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir()
#     nlp.to_disk(output_dir)
#     print("Saved model to", output_dir)

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

while True:
    print("Mời bạn nhâp văn bản:")
    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
    text = input()
    # text = "thời tiết Hà Nội hôm nay thế nào"
    print(text)
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# for text, _ in TRAIN_DATA:
#     doc = nlp2(text)
#     print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#     print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])