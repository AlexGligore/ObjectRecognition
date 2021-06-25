from scipy import ndimage
import skimage
from dataset_generation.src.config_manager import ConfigLoader
import numpy as np
import random
import cv2
import copy


class AugmentationOperations(object):
    def __init__(self, config_name="augmentation_ops_config.json"):
        self.config = ConfigLoader(config_path='./dataset_generation/config/',
                                   config_name=config_name).config
        self.available_ops = []
        self.parameters = {}
        self._set_available_ops()
        self._set_parameters()

    def _set_available_ops(self):
        if self.config['grayscale'] is True:
            self.available_ops.append(self.img_grayscale)
        if self.config['flip'] is True:
            self.available_ops.append(self.img_flip)
        if self.config['rotate'] is True:
            self.available_ops.append(self.img_rotate)
        if self.config['shift'] is True:
            self.available_ops.append(self.img_shift)
        if self.config['noise'] is True:
            self.available_ops.append(self.img_noise)
        if self.config['blur'] is True:
            self.available_ops.append(self.img_blur)

        if len(self.available_ops) == 0:
            raise ValueError("No operations allowed, please check config file for augmentation ops.")

    def _set_parameters(self):
        self.parameters["rotation_values"] = self.config["rotation_values"]
        self.parameters["shifting_values"] = self.config["shifting_values"]
        self.parameters["noise_values"] = self.config["noise_values"]
        self.parameters["blur_values"] = self.config["blur_values"]

    # img grayscale
    def img_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img flip
    def img_flip(self, img):

        flip = {
            0: np.flip,
            1: np.flipud,
            2: np.fliplr
        }

        operation = random.randint(0, 2)
        flipped_img = flip[operation](img)
        return flipped_img

    # img rotation
    def img_rotate(self, img):
        random_angle = self.parameters["rotation_values"]["random_angle"]
        specific_angle = self.parameters["rotation_values"]["specific_angle"]

        if random_angle is False and specific_angle is None:
            rotation = {
                0: cv2.ROTATE_90_CLOCKWISE,
                1: cv2.ROTATE_90_COUNTERCLOCKWISE,
                2: cv2.ROTATE_180
            }
            operation = random.randint(0, 2)
            rotated_img = cv2.rotate(img, rotation[operation])
        elif random_angle is True:
            angle = random.randint(0, 360)
            rotated_img = ndimage.rotate(img, angle)
        elif specific_angle is not None:
            angle = int(specific_angle)
            rotated_img = ndimage.rotate(img, angle)
        else:
            print("No rotation could be done to the image, please check the parameters")
            return img

        return rotated_img

    # img shifting
    def img_shift(self, img):
        min_shift = self.parameters["shifting_values"]["min_shift"]
        max_shift = self.parameters["shifting_values"]["max_shift"]

        # RGB img
        if len(img.shape) == 3:
            shift = [random.uniform(min_shift, max_shift), random.uniform(min_shift, max_shift), 0]
            shifted_img = ndimage.shift(img, shift)
        # Grayscale img
        else:
            shift = [random.uniform(min_shift, max_shift), random.uniform(min_shift, max_shift)]
            shifted_img = ndimage.shift(img, shift)

        return shifted_img

    # adding noise
    def img_noise(self, img):
        mean = self.parameters["noise_values"]["mean"]
        var = self.parameters["noise_values"]["var"]

        modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
        mode = modes[random.randint(0, len(modes) - 1)]

        if mode == 'gaussian' or mode == 'speckle':
            noisy_img = skimage.util.random_noise(img, mean=mean, var=var)
        else:
            noisy_img = skimage.util.random_noise(img)

        return np.asarray((255 * noisy_img)).astype(np.uint8)

    # blurring
    def img_blur(self, img):
        sigma = self.parameters["blur_values"]["sigma"]

        blurring_value = random.uniform(0, sigma)
        blurred_img = ndimage.gaussian_filter(img, sigma=blurring_value)
        return blurred_img


class GenerateDataset(object):
    def __init__(self, dataset_size, save_path, nr_of_operation_per_image=3):
        self.dataset_size = dataset_size
        self.save_path = save_path
        self.nr_op = nr_of_operation_per_image
        self.nr_possible_op = 5
        self.operations = AugmentationOperations()

    def generate_dataset(self, img, image_prefix=None, save=False):
        dataset = []
        for i in range(self.dataset_size):
            augmented_img = copy.copy(img)
            for j in range(self.nr_op):
                operation = random.randint(0, len(self.operations.available_ops) - 1)
                augmented_img = self.operations.available_ops[operation](augmented_img)

            if save is True:
                if image_prefix is not None:
                    img_name = self.save_path + str(image_prefix) + '_augmented_' + str(i) + '.jpg'
                else:
                    img_name = self.save_path + 'augmented_' + str(i) + '.jpg'
                cv2.imwrite(img_name, augmented_img)

            dataset.append(augmented_img)

        return dataset


# if __name__ == '__main__':
#     placeholder = np.zeros([255, 255, 3], dtype=np.uint8)
#     placeholder.fill(0)
#     print(img_flip(placeholder).shape)
#     print(img_rotate(placeholder).shape)
#     print(img_shift(img_grayscale(placeholder)).shape)
#     print(img_shift(placeholder).shape)
#     print(img_noise(placeholder).shape)
#     print(img_blur(placeholder).shape)
#     cv2.imshow('test', placeholder)
#     cv2.waitKey(0)
