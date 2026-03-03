from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class ROD_Dataset(CocoDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        (    "Pedestrian","Car","Cyclist","Tram","Truck"),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
