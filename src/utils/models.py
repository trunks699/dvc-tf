import tensorflow as tf
import os
import logging
from tensorflow.python.keras.backend import flatten
from src.utils.all_utils import get_timestamp
import glob
def get_VGG_16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"VGG16 base model saved at: {model_path}")
    return model

def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[-freeze_till:]:
            layer.trainable = False

    ## add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    logging.info("custom model is compiled and ready to be trained")
    full_model.summary()
    return full_model


def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from: {untrained_full_model_path}")
    return model

def get_unique_path_to_save_model(trained_model_dir, model_name="model_v*"):
    #timestamp = get_timestamp(model_name)
    ver = 0
    model_path=os.path.join(trained_model_dir,model_name)
    models = glob.glob(model_path)
    
    
    versions=[]
    
    new_model_name=""
    if(len(models)==0):
        new_model_name = model_name[:-1]
        new_model_name = new_model_name+(str(ver))
    else:
        for model in models:
            substr = model.split('_')
            substr=substr[1].split('.')
            s = substr[0]
            version = int(s[1:])
            versions.append(version)
        last_version = max(versions)
        new_version = int(last_version) + 1
        new_model_name = model_name[:-1]
        new_model_name = new_model_name+(str(new_version))
    unique_model_name = f"{new_model_name}.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    
    return unique_model_path