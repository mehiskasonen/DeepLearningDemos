from tensorflow import keras
import numpy as np

"""
    Part 4. Making a single prediction.
    np.expand_dims adds an extra dimension to the input data, corresponding to the batch dimension on the training set.
    Dimension added to the patch will be the first dimension, so axis is 0.
    Result also has a batch, so result on index [0][0] corresponds to the batch dimension of the input.
"""

def predict_single_image():

    test_image = keras.utils.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    cnn = keras.models.load_model('trained_model.h5')
    result = cnn.predict(test_image/255.0)

    # Get class names from the dataset
    class_names = ['cat', 'dog']
    # class_names = train_dataset.class_names
    print(f"Class names: {class_names}")

    # Create a dictionary mapping class names to their indices
    class_indices = {name: index for index, name in enumerate(class_names)}
    print(f"Class indices: {class_indices}")

    if result[0][0] > 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)


if __name__ == '__main__':
    predict_single_image()
