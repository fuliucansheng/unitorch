# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# Define the default palette using a dictionary comprehension
default = {c: (c // 64 * 32, c // 8 % 8 * 32, c % 8 * 32) for c in range(512)}

# Define the PASCAL palette as a list of tuples
_PASCAL = [
    #  class name           id   train   color
    ("background", 0, 0, (0, 0, 0)),
    ("aeroplane", 1, 1, (128, 0, 0)),
    ("bicycle", 2, 2, (0, 128, 0)),
    ("bird", 3, 3, (128, 128, 0)),
    ("boat", 4, 4, (0, 0, 128)),
    ("bottle", 5, 5, (128, 0, 128)),
    ("bus", 6, 6, (0, 128, 128)),
    ("car", 7, 7, (128, 128, 128)),
    ("cat", 8, 8, (64, 0, 0)),
    ("chair", 9, 9, (192, 0, 0)),
    ("cow", 10, 10, (64, 128, 0)),
    ("diningtable", 11, 11, (192, 128, 0)),
    ("dog", 12, 12, (64, 0, 128)),
    ("horse", 13, 13, (192, 0, 128)),
    ("motorbike", 14, 14, (64, 128, 128)),
    ("person", 15, 15, (192, 128, 128)),
    ("pottedplant", 16, 16, (0, 64, 0)),
    ("sheep", 17, 17, (128, 64, 0)),
    ("sofa", 18, 18, (0, 192, 0)),
    ("train", 19, 19, (128, 192, 0)),
    ("tvmonitor", 20, 20, (0, 64, 128)),
    ("borderingregion", 255, 21, (224, 224, 192)),
]

# Create the PASCAL palette dictionary from the _PASCAL list of tuples
pascal = {p[2]: p[3] for p in _PASCAL}

# Define the COCO palette as a list of tuples
_COCO = [
    #  class name           id   train   color
    ("background", 0, 0, (0, 0, 0)),
    ("person", 1, 1, (220, 20, 60)),
    ("bicycle", 2, 2, (119, 11, 32)),
    ("car", 3, 3, (0, 0, 142)),
    ("motorcycle", 4, 4, (0, 0, 230)),
    ("airplane", 5, 5, (0, 60, 100)),
    ("bus", 6, 6, (0, 0, 230)),
    ("train", 7, 7, (0, 80, 100)),
    ("truck", 8, 8, (0, 0, 70)),
    ("boat", 9, 9, (0, 0, 230)),
    ("traffic light", 10, 10, (250, 170, 30)),
    ("fire hydrant", 11, 11, (250, 170, 160)),
    ("stop sign", 12, 12, (220, 220, 0)),
    ("parking meter", 13, 13, (192, 192, 192)),
    ("bench", 14, 14, (64, 64, 128)),
    ("bird", 15, 15, (128, 64, 128)),
    ("cat", 16, 16, (64, 0, 64)),
    ("dog", 17, 17, (192, 0, 128)),
    ("horse", 18, 18, (64, 64, 0)),
    ("sheep", 19, 19, (128, 128, 0)),
    ("cow", 20, 20, (192, 192, 0)),
    ("elephant", 21, 21, (64, 128, 192)),
    ("bear", 22, 22, (192, 128, 192)),
    ("zebra", 23, 23, (64, 64, 128)),
    ("giraffe", 24, 24, (192, 192, 128)),
    ("hat", 25, 25, (0, 0, 192)),
    ("backpack", 26, 26, (0, 128, 192)),
    ("umbrella", 27, 27, (0, 128, 64)),
    ("shoe", 28, 28, (128, 128, 64)),
    ("eye glasses", 29, 29, (192, 128, 64)),
    ("handbag", 30, 30, (64, 0, 192)),
    ("tie", 31, 31, (192, 0, 64)),
    ("suitcase", 32, 32, (128, 0, 192)),
    ("frisbee", 33, 33, (64, 128, 64)),
    ("skis", 34, 34, (192, 128, 64)),
    ("snowboard", 35, 35, (0, 0, 255)),
    ("sports ball", 36, 36, (64, 255, 0)),
    ("kite", 37, 37, (192, 255, 0)),
    ("baseball bat", 38, 38, (64, 64, 0)),
    ("baseball glove", 39, 39, (192, 64, 0)),
    ("skateboard", 40, 40, (64, 192, 0)),
    ("surfboard", 41, 41, (192, 192, 0)),
    ("tennis racket", 42, 42, (64, 64, 128)),
    ("bottle", 43, 43, (192, 64, 128)),
    ("wine glass", 44, 44, (0, 0, 255)),
    ("cup", 45, 45, (64, 255, 255)),
    ("fork", 46, 46, (192, 255, 255)),
    ("knife", 47, 47, (64, 0, 255)),
    ("spoon", 48, 48, (192, 0, 255)),
    ("bowl", 49, 49, (0, 255, 255)),
    ("banana", 50, 50, (64, 255, 0)),
    ("apple", 51, 51, (192, 255, 0)),
    ("sandwich", 52, 52, (64, 0, 255)),
    ("orange", 53, 53, (192, 0, 255)),
    ("broccoli", 54, 54, (0, 255, 128)),
    ("carrot", 55, 55, (64, 255, 128)),
    ("hot dog", 56, 56, (192, 255, 128)),
    ("pizza", 57, 57, (64, 0, 128)),
    ("donut", 58, 58, (192, 0, 128)),
    ("cake", 59, 59, (0, 128, 255)),
    ("chair", 60, 60, (64, 128, 255)),
    ("couch", 61, 61, (192, 128, 255)),
    ("potted plant", 62, 62, (0, 0, 64)),
    ("bed", 63, 63, (128, 0, 64)),
    ("dining table", 64, 64, (0, 128, 64)),
    ("toilet", 65, 65, (128, 128, 64)),
    ("tv", 66, 66, (0, 0, 0)),
    ("laptop", 67, 67, (128, 0, 64)),
    ("mouse", 68, 68, (0, 128, 128)),
    ("remote", 69, 69, (128, 128, 128)),
    ("keyboard", 70, 70, (64, 0, 64)),
    ("cell phone", 71, 71, (192, 0, 64)),
    ("microwave", 72, 72, (64, 128, 64)),
    ("oven", 73, 73, (192, 128, 64)),
    ("toaster", 74, 74, (64, 0, 128)),
    ("sink", 75, 75, (192, 0, 128)),
    ("refrigerator", 76, 76, (64, 128, 128)),
    ("blender", 77, 77, (192, 128, 128)),
    ("book", 78, 78, (0, 64, 0)),
    ("clock", 79, 79, (128, 64, 0)),
    ("vase", 80, 80, (0, 192, 0)),
    ("scissors", 81, 81, (128, 192, 0)),
    ("teddy bear", 82, 82, (0, 64, 128)),
    ("hair drier", 83, 83, (128, 64, 128)),
    ("toothbrush", 84, 84, (0, 192, 128)),
]

# Create the COCO palette dictionary from the _COCO list of tuples
coco = {p[2]: p[3] for p in _COCO}

# Create a dictionary that maps palette names to the corresponding palette
palette = {
    "default": default,
    "pascal": pascal,
    "coco": coco,
}


def get(name):
    assert name in palette
    return palette.get(name)
