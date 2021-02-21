from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

from keras.applications.vgg16 import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

size=[224,224]
train=('Train')
test=('Test')

VGG = VGG16(input_shape=size + [3],weights='imagenet',include_top=False)  #using VGG 16 transfer learning model
for layer in VGG.layers:
    layer.trainable = False
folders=glob("Train//*")
x=Flatten()(VGG.output)                                              #returning a 1 D array of the output
prediction=Dense(len(folders),activation='softmax')(x)               #softmac activation is used
model = Model(inputs=VGG.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
batch_size=32
train_datagen=ImageDataGenerator(rescale = 1./255,                      #using image augmentation for training dataset
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip=True)
test_datagen=ImageDataGenerator(rescale=1./225,horizontal_flip=True,vertical_flip=True)  #using image augmentation on the testing dataset

train_set=train_datagen.flow_from_directory("Train",
                                            batch_size=32,
                                            target_size=[224,224],
                                            class_mode='categorical')
test_set=test_datagen.flow_from_directory("Test",
                                          batch_size=32,
                                          target_size=[224,224],
                                          class_mode='categorical')
history = model.fit(                                                          #training the model
  train_set,
  validation_data=test_set,
  epochs=25,
  steps_per_epoch=len(train_set)//batch_size,
  validation_steps=len(test_set)
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save("model.h5")