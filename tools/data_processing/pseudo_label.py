import argparse 
import os.path as osp
import os
import re 
import shutil
from gencls.engine.engine import Engine
from tools.misc.config import Cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for pseudo-label')
    parser.add_argument('--root_dir', type=str, required=True, help='Directionary path of images')
    parser.add_argument('--config', type=str, required=True, help="config path")
    parser.add_argument('--load_from', type=str, required=True,  help='weight path')
    parser.add_argument('--out_dir', type=str, required=True, help="Output dir")
    parser.add_argument('--threshold', type=float, help='threshold for confuse case')
    args = parser.parse_args()

    config = Cfg.load_config_from_file(args.config)
    
    class_dict = config['Infer']['PostProcess']['class_id_mapping']
    # prepare output dir
    for class_name in class_dict.keys():
        outpath = osp.join(args.out_dir, class_name)
        if not osp.exists(outpath):
            os.makedirs(outpath, exist_ok=True)
    confuse_dir = osp.join(args.out_dir, "confuse")
    if not osp.exists(confuse_dir):
        os.makedirs(confuse_dir, exist_ok=True)
    
    config['Infer']['img'] = args.root_dir
    config['Common']['pretrained_model'] = args.load_from
    print(Cfg.pretty_text(config))
    engine = Engine(config=config, mode='pseudo')

    results = engine.infer_on_dataloader()
    # expect results format: 
    '''{
        'class_ids': [0], 
        'scores': [1.0], 
        'file_name': '../EHR/ehrs_search_engine/my_test/crop_img_misc_hw_vis\\133709471_698438557726722_3692624307849311084_n_0.jpg', 
        'label_name_list': ['print']
    }
    
    '''
    for item in results:
        scores = item['scores'][0]
        # print(scores)
        if scores <= args.threshold:
            out_dir = confuse_dir
        else:
            out_dir = osp.join(args.out_dir, item['label_name_list'][0])
        img_path  = item['file_name']
        shutil.copy(img_path, out_dir)


