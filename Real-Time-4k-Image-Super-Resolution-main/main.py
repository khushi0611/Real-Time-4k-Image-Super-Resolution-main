# pip install tensorflow
# pip install tensorflow-hub
# pip install opencv-python
# pip install numpy
# pip install matplotlib


import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('testCase2.jpg')

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not load the image.")
    exit()

# Check if the loaded image has non-zero dimensions
if img.size == 0:
    print("Error: The loaded image is empty.")
    exit()

# Check if the image is in BGR format
if len(img.shape) != 3 or img.shape[2] != 3:
    print("Error: Image is not in BGR format.")
    exit()

# Display the original image
image_plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image_plot)
plt.title("Image")
plt.show()

# Model to preprocess the images
def preprocessing(img):
    image_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, image_size[0], image_size[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32)
    return tf.expand_dims(preprocessed_image, 0)

# Load the ESRGAN model from TensorFlow Hub
esrgan_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
esrgan_model = hub.load(esrgan_path)

# Super-resolution model function
def sr_model(img):
    preprocessed_image = preprocessing(img)
    new_image = esrgan_model(preprocessed_image) / 255.0
    return tf.squeeze(new_image)

# Display the original and super-resolved images
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.title(" Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display the super-resolved image
hr_image = sr_model(img).numpy()
hr_image = (hr_image * 255).astype(np.uint8)

plt.subplot(1, 2, 2)
plt.title("Super-Resoluted Image")
plt.imshow(hr_image)
plt.axis('off')

plt.show()
