import cv2
import os
from glob import glob

# use your skimage scaling
def preprocess_images(input_dir, output_lr_dir, output_hr_dir, scale=4):
    print("Making Directories")
    os.makedirs(output_lr_dir, exist_ok=True)
    os.makedirs(output_hr_dir, exist_ok=True)
    print("Finished Making Directories")

    for image_path in glob(os.path.join(input_dir, "*.png")):
        print(f"Processing Image: {image_path}")
        image = cv2.imread(image_path)
        high_res = cv2.resize(image, (image.shape[1], image.shape[0]))
        
        # scale down by 4
        low_res = cv2.resize(high_res, (high_res.shape[1] // scale, high_res.shape[0] // scale), interpolation=cv2.INTER_CUBIC)

        # scale back up
        low_res = cv2.resize(low_res, (high_res.shape[1], high_res.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Save the images
        hr_path = os.path.join(output_hr_dir, os.path.basename(image_path))
        lr_path = os.path.join(output_lr_dir, os.path.basename(image_path))
        print("Saved Image")
        
        cv2.imwrite(hr_path, high_res)
        cv2.imwrite(lr_path, low_res)

preprocess_images("unprocessed_images", "processed_train_lr", "processed_train_hr")