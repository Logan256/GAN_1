from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import cv2

# Load the saved model
def tensorf(x):
    return tf.image.resize(x, (256, 256))

model = load_model('generator_model.keras', compile=True, safe_mode=False)

# Preprocess the image (resize, normalize, etc.)
lr_image = cv2.imread("test_images/0801x4.png")
# lr_image = cv2.resize(lr_image, (lr_image.shape[1], lr_image.shape[0])) / 255.0

input_image_resized = cv2.resize(lr_image, (256, 256)) / 255.0  # Normalizing to [0, 1]
input_image_resized = np.expand_dims(input_image_resized, axis=0)  # Adding batch dimension

# Make prediction
output_image = model.predict(input_image_resized)

# Postprocess the output (if needed)
output_image = output_image[0]  # Removing the batch dimension
output_image = (output_image * 255).astype(np.uint8)  # Scaling back to [0, 255]

# Display or save the output image
# cv2.imshow('Predicted Image', output_image)
cv2.imwrite('predicted_image.png', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



