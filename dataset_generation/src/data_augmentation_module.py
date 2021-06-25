import cv2
import os
from dataset_generation.src.augmentation_ops import GenerateDataset


class AugmentDataModule(object):
    def __init__(self):
        pass

    @staticmethod
    def read_images(data_folder, img_format):
        # read the images

        image_data = {}
        for img in os.listdir(data_folder):
            read_img_format = os.path.splitext(img)[1]
            if read_img_format in img_format:
                image_data[img] = cv2.imread(data_folder + img, cv2.IMREAD_COLOR)
        print("Image reading complete.")
        return image_data

    @staticmethod
    def resize_images(image_data, images_size, save=False, save_path=None):
        # resize the images
        resized_images = {}
        for img in image_data:
            resized_images[img] = cv2.resize(image_data[img], images_size, interpolation=cv2.INTER_AREA)

        # save resized images
        if save is True:
            if os.path.exists(save_path) is False:
                os.makedirs(save_path, exist_ok=True)

            for img in resized_images:
                path = save_path + img
                cv2.imwrite(path, resized_images[img])
        print("Image resizing complete.")

        return resized_images

    @staticmethod
    def augment_dataset(image_data, augmented_dataset_size_per_img=50, save=False, save_path=None):
        # augment a dataset
        complete_dataset = {}
        dataset_generator = GenerateDataset(augmented_dataset_size_per_img, save_path)
        for img in image_data:
            img_name = os.path.splitext(img)[0]
            if save is True:
                temp_save_path = save_path + img_name + '/'
                dataset_generator.save_path = temp_save_path
                if os.path.exists(temp_save_path) is False:
                    os.makedirs(temp_save_path, exist_ok=True)

            temp_dataset = dataset_generator.generate_dataset(image_data[img], image_prefix=img_name, save=save)
            complete_dataset[img_name] = temp_dataset

        if save is True:
            print("Data augmentation completed successfully. Results can be found in {0}".format(save_path))
            return complete_dataset
        else:
            print("Data augmentation completed successfully.")
            return complete_dataset


# if __name__ == '__main__':
#     dataset_augmentation = AugmentDataModule()
#     images_folder = './data/originals/'
#     images_format = '.jpg'
#     img_data = dataset_augmentation.read_images(images_folder, images_format)
#
#     resized_folder = './data/resized_images/'
#     resized_imgs = dataset_augmentation.resize_images(image_data=img_data, save=False, save_path=resized_folder)
#
#     augmented_dataset_save_path = './data/augmented_dataset/'
#     augmented_dataset = dataset_augmentation.augment_dataset(image_data=resized_imgs, save=False,
#                                                              save_path=augmented_dataset_save_path)
