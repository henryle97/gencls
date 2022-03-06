
from genericpath import exists
import glob
import argparse
import os 
import os.path as osp 

class_dict = {
    'print': 0,
    'handwriting': 1
}


def create_txt_dataset(data_dir, txt_out):
    paths = glob.glob(data_dir + "/*/*.jpg")
    print("num images: ", len(paths))
    with open(txt_out, 'w', encoding='utf8') as f:
        for path in paths:
            path = path.replace("\\", "/")
            name_class = path.split("/")[-2]
            if name_class not in class_dict:
                continue
            class_id = class_dict[name_class]
            f.write(path + "\t" + str(class_id) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument parser for create file label')
    parser.add_argument('--image_dir', type=str,
                        required=True, help="image dir path")
    parser.add_argument('--txt_out', type=str,
                        required=True, help="txt_out path")
    args = parser.parse_args()
    save_dir = osp.dirname(args.txt_out)
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    create_txt_dataset(args.image_dir, args.txt_out)
