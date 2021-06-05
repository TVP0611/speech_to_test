
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.sputils import matrix
# %matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
import pickle

def convert_num2label(num):
    convert_num2label = {0: 'bật', 1: 'đóng', 2: 'mở', 3: 'ngắt', 4: 'off', 5: 'on', 6: 'tắt'}
    return convert_num2label[num]

def load_image_files(container_path, dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def get_test_image(path, dimension=(64, 64)):
    images = []
    flat_data = []
    img = imread(path)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    # images.append(img_resized)
    flat_data.append(img_resized.flatten()) 
    # images = np.array(images)
    flat_data = np.array(flat_data)
    return flat_data

image_dataset = load_image_files("D:/Project/process_voice/process_voice_classify/save_spec/")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

#################   train model
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
# svc = svm.SVC()
# clf = GridSearchCV(svc, param_grid)
# clf.fit(X_train, y_train)
# ##################  Save model
# pickle.dump(clf, open("model_svm_audio.pk", 'wb'))

##################   load model
clf = pickle.load(open("D:/Project/process_voice/process_voice_classify/model_svm_audio.pk", 'rb'))
# image_test = load_image_files
# y_pred = clf.predict(X_test)
# # print(y_pred)
# # print(y_test)

# print("Classification report for - \n{}:\n{}\n".format(
#     clf, metrics.classification_report(y_test, y_pred)))

# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

image = get_test_image("D:/Project/process_voice/process_voice_classify/test_spec/bat/bat_0.png")
y_result = clf.predict(image)
print(y_result)
print(convert_num2label(y_result[0]))
