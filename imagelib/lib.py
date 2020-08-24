"""
Image lib
@Author Chanwoo Kwon, Yonsei University Researcher, 2020.05~
"""

import cv2
import os


class Extractor:
    """
    vector extractor
    extract vector value from graph image
    """
    def __init__(self, base_path):
        """
        initializer
        :param base_path: base path of image file list
        """
        self.base_path = base_path

    def extract(self):
        """
        extract vector value from graph image
        :return: array of image path & vector pairs
        """
        file_list = os.listdir(self.base_path)
        vector_list = []
        for image_file in file_list:
            try:
                image_mat = cv2.imread(image_file)
                vector_list.append({"name": image_file, "vector": self.__get_vector(image_mat)})
            except:  # if image_file is not image
                pass

    def __get_vector(self, image_mat):
        """
        read y-value, x-value from image matrix using opencv lib
        :param image_mat: image matrix read by opencv lib
        :return: image y-values vector
        """
        pass
