import argparse
from gencls.engine.engine import Engine
from tools.misc.config import Cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument parser for pseudo-label')
    parser.add_argument('--config', type=str,
                        required=True, help="config path")
    parser.add_argument('--resume_from', type=str,
                        required=False, help="config path")
    parser.add_argument('--pretrained', type=str,
                        required=False, help="config path")
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    config = Cfg.update_from_args(config, args)
    
    engine = Engine(config, mode='train')
    engine.train()