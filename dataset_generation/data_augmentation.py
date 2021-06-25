from dataset_generation.src.config_manager import ConfigLoader
from dataset_generation.src.data_augmentation_module import AugmentDataModule
import cv2


class DataAugmentation(object):
    def __init__(self, config_path=None, config_name=None):
        if config_path is not None and config_name is not None:
            self.config = ConfigLoader(config_path, config_name).config
        else:
            self.config = ConfigLoader().config

    def augment_dataset(self):
        img_data = AugmentDataModule.read_images(self.config['images_folder'], self.config['images_format'])
        img_size = tuple(self.config['images_size'])
        # initial resize
        resized_imgs = AugmentDataModule.resize_images(img_data, img_size, save=False)
        augmented_imgs = AugmentDataModule.augment_dataset(resized_imgs,
                                                           augmented_dataset_size_per_img=self.config['augmented_dataset_size_per_img'],
                                                           save=self.config['save_results'],
                                                           save_path=self.config['augmented_dataset_folder'])

        # final_resize
        resized_imgs = {}
        for img_type in augmented_imgs.keys():
            resized_imgs[img_type] = []
            for img in augmented_imgs[img_type]:
                resized_img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                resized_imgs[img_type].append(resized_img)

        return resized_imgs


if __name__ == '__main__':
    # test
    data_augmentation = DataAugmentation()
    augmented_dataset = data_augmentation.augment_dataset()

    for key, value in augmented_dataset.items():
        print("In dataset for {0}, we augmented {1} samples.".format(key, len(value)))
