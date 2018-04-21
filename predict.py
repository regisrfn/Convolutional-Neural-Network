# prediction on a new picture
from keras.preprocessing import image as image_utils
from keras.models import model_from_json
import numpy as np

test_image = image_utils.load_img('/home/regis/Documents/downloads/cats/12-cat.png', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

class_labels = np.array(['cat','dog'])
result = loaded_model.predict_on_batch(test_image)
print(result)
print(class_labels[int(np.round(result))])
