import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 

class RU(tf.keras.layers.Layer) :
    def __init__(self,filters,activation = "relu",strides = 1) :
        super(RU,self).__init__()
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters,3,strides = strides,
            padding = "same",use_bias = False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters,3,strides = 1,padding = "same",use_bias = False),
            tf.keras.layers.BatchNormalization(),
        ]

        self.skip_layers = []
        if strides > 1 :
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters,1,strides = strides,padding = "same",use_bias = False),
                tf.keras.layers.BatchNormalization()

            ]

    def call(self,inputs) :

        Z = inputs
        for Layer in self.main_layers :
            Z = Layer(Z)
        
        skip_Z = inputs
        for layer in self.skip_layers :
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,7,strides = 2,input_shape = [224,224,3],padding = 'same',use_bias = False))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size = 3,strides = 2,padding="same"))
pre_strides = 64

for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3 :
    strides = 1 if filters == pre_strides else 2
    model.add(RU(filters,strides=strides))
    pre_strides = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation = "softmax"))

#model.compile(optimizer='adam',loss='mse')
#model.fit(image,epochs=10)
model.summary() 


