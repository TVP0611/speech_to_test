import spacy
nlp=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner = nlp.get_pipe("ner")

# training data
TRAIN_DATA = [
              # ("Thành phố Hồ Chí Minh chỉ đạo phòng chống dịch", {"entities": [(0, 19, "LOC")]}),
              # ("Để đối phó với dịch Hải Phòng quyết tâm đến cùng", {"entities": [(20, 29, "LOC")]}),
              # ("Đà Nẵng quyết tâm đối phó với dịch bệnh", {"entities": [(0,6, "LOC")]}),
              # ("Toàn dân Cần Thơ quyết tâm chống dịch", {"entities": [(9,14, "LOC")]}),
              # ("Hôm nay Cà Mau khá nóng", {"entities": [(8,13, "LOC")]}),
              # ("Toàn dân tộc quyết tâm bảo vệ Trường Sa ", {"entities": [(30,38, "LOC")]}),
              # ("Hôm nay cả nhà tôi đi Sa Pa", {"entities": [(22,26, "LOC")]})

              ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
              ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
              ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
              ("I was driving a BMW", {"entities": [(16,19, "PRODUCT")]}),
              ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
              ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
              ("I bought a new Washer", {"entities": [(16,22, "PRODUCT")]}),
              ("I bought a old table", {"entities": [(16,21, "PRODUCT")]}),
              ("I bought a fancy dress", {"entities": [(18,23, "PRODUCT")]}),
              ("I rented a camera", {"entities": [(12,18, "PRODUCT")]}),
              ("I rented a tent for our trip", {"entities": [(12,16, "PRODUCT")]}),
              ("I rented a screwdriver from our neighbour", {"entities": [(12,22, "PRODUCT")]}),
              ("I repaired my computer", {"entities": [(15,23, "PRODUCT")]}),
              ("I got my clock fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("I got my truck fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
              ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Swiggy", {"entities": [(24,29, "ORG")]})
              ]

# Adding labels to the `ner`
for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(30):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=8)#compounding(4.0, 32.0, 1.001)
    for batch in batches:
        # texts, annotations = zip(*batch)
        for text, annotations in batch:
            # create Example
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                        [example],  # batch of texts
                        # annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses)
            print("Losses", losses)

# Testing the model
doc = nlp("I was driving a Alto")##Đà Nẵng quyết tâm đối phó với dịch bệnh      Đà Nẵng là một thành phố đáng sống
# for ent in doc.ents:
#     print(ent)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
