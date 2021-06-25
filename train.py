from architecture import simple_classifier
from dataset_generation.data_augmentation import DataAugmentation
from keras.utils.np_utils import to_categorical
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# generate a dataset
config_path = './dataset_generation/config/'
config_name = 'config.json'
data_augmentation = DataAugmentation(config_path=config_path, config_name=config_name)
augmented_dataset = data_augmentation.augment_dataset()

# create train data
labels = {}
label_index = 0
for key in augmented_dataset.keys():
    labels[key] = label_index
    label_index += 1

classes_nr = len(list(labels.keys()))

train_data = []
label_data = []

for key, value in augmented_dataset.items():
    label_value = labels[key]
    for img in value:
        train_data.append(img)
        label_data.append(label_value)

train_data = np.asarray(train_data).astype(np.float32)
label_data = np.asarray(label_data).astype(np.float32)
label_data = to_categorical(label_data, classes_nr)

model = simple_classifier(classes_nr)
model.fit(train_data, label_data, batch_size=4, epochs=100, verbose=1, shuffle=True)
model.save('test_model.h5')