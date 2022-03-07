from gencls.dataset import preprocess
def create_operators(params):
    """ create operators based on config 
    transforms:
        - Resize:
            size: [48, 96]
            keep_ratio: true

        - Normalize: 
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]   # rgb
            to_rgb: true
    Args:
        params (list[dict]): a dict list, used to create operators
    """
    assert isinstance(params, list)
    ops = []
    for operator in params:
        assert isinstance(operator, dict) and len(operator) == 1, 'yaml format error'
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(preprocess, op_name)(**param)
        ops.append(op)
        print(ops)
    return ops