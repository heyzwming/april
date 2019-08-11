import cv2
import numpy as np
from os import walk
from os.path import join

def create_descriptors(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        if '.jpg' in f:
            save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())

def save_descriptor(folder, image_path, feature_detector):
    # ÅÐ¶ÏÍŒÆ¬ÊÇ·ñÎªnpyžñÊœ
    if image_path.endswith("npy"):
        return
    # ¶ÁÈ¡ÍŒÆ¬²¢Œì²éÌØÕ÷
    img = cv2.imread(join(folder,image_path), 0)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    # ÉèÖÃÎÄŒþÃû²¢œ«ÌØÕ÷ÊýŸÝ±£ŽæµœnpyÎÄŒþ
    descriptor_file = image_path.replace("jpg", "npy")
    np.save(join(folder, descriptor_file), descriptors)

if __name__=='__main__':
    image_path = 'E:\\anchors'
    create_descriptors(image_path)

