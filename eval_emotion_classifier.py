import os
import glob
import re
from keras.models import load_model

from data import load_emotion_data, split_data
import time

# get all the models with a validation accuracy > 0.6
model_files = glob.glob('models/*.hdf5')
other_models = glob.glob('models/*/*-0.6*hdf5')
model_files.extend(other_models)

public_test_dict = {}
private_test_dict = {}
results = {}

for model_file in model_files:
    model = load_model(model_file, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    input_shape = model.input_shape[1:4]
    if input_shape not in public_test_dict.keys():
        faces, emotions = load_emotion_data('data/fer2013/fer2013.csv', input_shape)
        train, test = split_data(faces, emotions, 0.2)
        public_test_dict[input_shape], private_test_dict[input_shape] = split_data(test[0], test[1], 0.5)


    start = time.time()
    public_test_result = model.evaluate(public_test_dict[input_shape][0], public_test_dict[input_shape][1])
    private_test_result = model.evaluate(private_test_dict[input_shape][0], private_test_dict[input_shape][1])
    duration = time.time() - start
    print(model_file)
    print('public  test', public_test_result)
    print('private test', private_test_result)
    results[model_file] = {'public_acc': public_test_result[1], 'private_acc': private_test_result[1], 'time': duration}
print(results)
import json
json.dump(results, open('test.json', 'w'))
