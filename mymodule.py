from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

def architecture1(model,input_shape,num_classes):
 model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
 model.add(BatchNormalization())

 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(BatchNormalization())

 model.add(Dropout(0.5))
 model.add(Dense(num_classes, activation='softmax'))
 
 epochs = 1
 return epochs 	

def architecture2(model,input_shape,num_classes):
 model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
 model.add(BatchNormalization())

 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(BatchNormalization())

 model.add(Dropout(0.5))
 model.add(Dense(num_classes, activation='softmax'))
 
 epochs = 3
 return epochs

def architecture3(model,input_shape,num_classes):
 model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
 model.add(BatchNormalization())

 model.add(Conv2D(64, (3, 3), activation='relu'))
 model.add(BatchNormalization())
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.25))

 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(BatchNormalization())

 model.add(Dropout(0.5))
 model.add(Dense(num_classes, activation='softmax'))	
 
 epocs = 1
 return epochs

def architecture4(model,input_shape,num_classes):
 model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
 model.add(BatchNormalization())

 model.add(Conv2D(64, (3, 3), activation='relu'))
 model.add(BatchNormalization())
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.25))

 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(BatchNormalization())

 model.add(Dropout(0.5))
 model.add(Dense(num_classes, activation='softmax'))	
 
 epochs = 3
 return epochs
