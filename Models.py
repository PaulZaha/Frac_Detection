import tensorflow as tf
import numpy as np


def model_sequential(train_generator,validation_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(64 , 64, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    #model.summary()
    model_compiler(model)
    model_fitter(model,train_generator)
    model_evaluater(model,validation_generator)

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)
    

def model_CNN(train_generator, validation_generator):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
    ])
    model_compiler(model)
    model_fitter(model,train_generator)
    model_evaluater(model,validation_generator)

    conf_matrix,correct_value_perc = boolean_conf_matrix(model,validation_generator)
    print(conf_matrix)
    print(correct_value_perc)



def model_compiler(model):
    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
def model_fitter(model,train_generator):
    model.fit(train_generator,epochs=1)

def model_evaluater(model,validation_generator):
    model.evaluate(validation_generator,verbose=1)

def boolean_conf_matrix(model,generator):
    true_labels = np.array(generator.classes)
    print(true_labels)
    predicted_labels_binary = (model.predict(generator)[:, 0] > 0.5).astype(int)
    print(predicted_labels_binary)
    conf_matrix = tf.math.confusion_matrix(true_labels,predicted_labels_binary)

    correct_value_perc = ((conf_matrix[0,0] + conf_matrix[1,1]) / np.sum(conf_matrix))

    return conf_matrix,correct_value_perc