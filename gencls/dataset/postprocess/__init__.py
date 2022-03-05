import copy
import importlib 
from .topk import Topk


def build_postprocess(config):
    config = copy.deepcopy(config)
    model_name = config.pop("name")
    mod = importlib.import_module(__name__)
    postprocess_func = getattr(mod, model_name)(**config)
    return postprocess_func


if __name__ == "__main__":
    build_postprocess({
        'name': 'Topk',
        'topk': 1
    })