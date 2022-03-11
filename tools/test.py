import argparse
from gencls.engine.engine import Engine
from tools.misc.config import Cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument parser for pseudo-label')
    parser.add_argument('--config', type=str,
                        required=True, help="config path")
    parser.add_argument('--pretrained', type=str,
                        required=True, help="config path")
    parser.add_argument('--metric', type=str,
                    required=True, help="accuracy | f1 | confusion")
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    config = Cfg.update_from_args(config, args)
    engine = Engine(config, mode='eval')
    result = engine.test()
    print(result)

    # python tools/test.py --config config/print_hw_cls.yml --pretrained weights/mv3_large_scale1_aug_color.pt --metric accuracy