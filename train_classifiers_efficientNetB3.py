# -*- coding: utf-8 -*-
"""
@author: Gifani
"""

# loading model based on imagenet or noisy_student




import efficientnet.keras as efn 
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
model_name='EfficientNetB3'




img_width, img_height = 300,300
num_channels = 3
train_data = './train'
valid_data = './valid'
model_path = './models/student/'
num_classes = 4
num_train_samples =1817
num_valid_samples =227
verbose = 1
batch_size = 4
num_epochs = 50
patience = 50

log_file_path = model_path + model_name +  '/training_'  + model_name + '.log'
trained_models_path = model_path + model_name + '/model_'

#base_model = efn.EfficientNetB0(input_shape=( 224,224,3), weights='imagenet', include_top=False)
base_model = efn.EfficientNetB3(input_shape=( 300,300,3), weights='noisy-student', include_top=False)


base_model.summary()
num_L=0
for lyr in base_model.layers:
    
    if lyr.trainable:
        #print(lyr.name)
        num_L = num_L +1

print(num_L)
x = keras.layers.AveragePooling2D((10,10))(base_model.output)


x_newfc = keras.layers.Flatten()(x)

x_newfc = keras.layers.Dense(num_classes, activation='softmax', name='fc_new')(x_newfc)

model = keras.models.Model(input=base_model.input, output=x_newfc)
model.summary()
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
rms= optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-6)
adam=optimizers.Adam( lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    
    # prepare data augmentation configuration
train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
valid_data_gen = ImageDataGenerator()
    # callbacks
tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)

model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}_' + str(num_classes) + 'class_' + model_name + '.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    # generators
train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')

    # fine tune the model
model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=valid_generator,
        validation_steps=num_valid_samples / batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose)


