import glob 
import os 

EXT_IMGS = ['.jpg', '.png', '.jpeg', '.tif']

def get_image_list(img_dir):
    img_paths = glob.glob(img_dir + "/*")
    img_paths = [img_path for img_path in img_paths if os.path.splitext(img_path)[-1] in EXT_IMGS] 
    return img_paths
