def transform(img, ops):
    for op in ops:
        img = op(img)
    return img