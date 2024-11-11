import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_cnn():

    """
    Part 1. Data preprocessing.
    """

    """Preprocessing the training set"""

    train_datagen = keras.Sequential([
        layers.Rescaling(scale=1. / 255),
        layers.RandomFlip('horizontal'),
        layers.RandomZoom(0.2),
        layers.RandomRotation(0.2)
    ])

    training_set = tf.keras.utils.image_dataset_from_directory('dataset/training_set',
                                                               image_size=(64, 64),
                                                               batch_size=32)

    # Apply the data augmentation to the training dataset
    train_dataset = training_set.map(
        lambda x, y: (train_datagen(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch the data for improved performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    """Preprocessing the test set"""
    test_datagen = keras.Sequential([
        layers.Rescaling(scale=1. / 255)
    ])

    test_set = tf.keras.utils.image_dataset_from_directory('dataset/test_set',
                                                           image_size=(64, 64),
                                                           batch_size=32)

    # Apply the rescaling to the test dataset
    test_dataset = test_set.map(
        lambda x, y: (test_datagen(x, training=False), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch the data for improved performance
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    """Part 2. Building the CNN."""

    """Initialising the CNN"""
    cnn = keras.models.Sequential()

    """Step 1. Convolution."""
    cnn.add(keras.Input(shape=(64, 64, 3)))
    cnn.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    """Step 2. Pooling."""
    cnn.add(layers.MaxPooling2D(pool_size=2, strides=2))

    """Adding a second convolutional layer"""
    cnn.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    cnn.add(layers.MaxPooling2D(pool_size=2, strides=2))

    """Step 3. Flattening."""
    cnn.add(layers.Flatten())

    """Step 4. Full connection layer."""
    cnn.add(layers.Dense(units=128, activation='relu'))

    """Step 5. Output layer."""
    cnn.add(layers.Dense(units=1, activation='sigmoid'))

    """
    Part 3. Training the CNN.
    """

    """Compiling the CNN"""
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """Training the CNN on the training set and evaluating the CNN on Test set."""
    cnn.fit(x=train_dataset, validation_data=test_set, epochs=25)

    """Saving CNN model"""
    cnn.save('trained_model')


if __name__ == '__main__':
    train_cnn()
