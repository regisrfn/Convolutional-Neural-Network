from keras.models import model_from_json

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 128,
                                            class_mode = 'categorical')

y_test = test_set.classes

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
# Compiling the CNN
loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
score = loaded_model.evaluate_generator(test_set)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# y_pred = loaded_model.predict_generator(test_set)
# y_pred = y_pred.argmax(axis=1)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred, labels = [0,1])