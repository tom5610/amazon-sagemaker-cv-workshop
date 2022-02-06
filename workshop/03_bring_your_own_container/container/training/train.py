from __future__ import absolute_import, division, print_function

import argparse
import logging
import ast
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

HEIGHT = 32
WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 10

BATCH_SIZE = 32

def _get_callbacks():
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        mode = 'min',
        restore_best_weights = True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience = 2, mode = 'min', verbose=1)
    
    return [early_stopping, reduce_lr]

def _get_datasets(training_dir, validation_dir, batch_size):
    
    train_data_generator = ImageDataGenerator(
        rescale = 1.0 / 255.0, 
        validation_split = 0.0, # it's for training dataset only
        shear_range = 0.2,
        zoom_range =0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
    )

    # training dataset
    train_data_multi = train_data_generator.flow_from_directory(
        directory = training_dir,
        target_size = (HEIGHT, WIDTH),
        class_mode = 'categorical',
        batch_size = batch_size,
        shuffle = True,
        seed = 42
    )

    # testing data generator
    validation_data_generator = ImageDataGenerator(
        rescale = 1.0 / 255.0
    )

    # testing dataset
    validation_data_multi = validation_data_generator.flow_from_directory(
        directory = validation_dir,
        target_size = (HEIGHT, WIDTH),
        class_mode = 'categorical',
        batch_size = batch_size,
        shuffle = True,
        seed = 42
    )
    
    return train_data_multi, validation_data_multi
    

def _train(args):
    
    tf.random.set_seed(99)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (HEIGHT, WIDTH, CHANNELS)),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(args.num_classes, activation = 'softmax')
    ])
    
    opt = tf.keras.optimizers.RMSprop(learning_rate = 0.0001, decay=1e-6)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    print(model.summary())
    
    callbacks = _get_callbacks()
    
    # fit the training data
    training_data, validation_data = _get_datasets(args.training, args.validation, args.batch_size)
    
    history = model.fit(
        training_data, 
        batch_size = args.batch_size,
        epochs = args.epochs,
        validation_data = validation_data,
        verbose = 2,
        callbacks = callbacks
    )
    
    # output the last epoch metrics: validation loss & specified metrics
    metric_names = list(history.history)
    print(
        "Training results: "
        + "; ".join(map(
            lambda i: f"{metric_names[i]}={history.history[metric_names[i]][-1]:.5f}", range(len(metric_names))
        ))
    )
    
    ###### Save Keras model for TensorFlow Serving ############
    export_path = f"{args.model_dir}/1"

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="E",
        help="number of total epochs to run (default:10)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="BS",
        help="batch size (default:32)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="initial learning rate (default:0.0001)"
    )
    
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-6,
        metavar="D",
        help="weight decay (default:1e-6)"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        metavar="NC",
        help="number of classes (default:10)"
    )
    
    # to retrieve default values from SageMaker environment variables, which will be instantiated by SageMaker containers framework
    parser.add_argument("--host", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    _train(parser.parse_args())