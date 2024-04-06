from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DaconDataset(CustomDataset):

    CLASSES=('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole', 'Traffic Light', 'Traffic Sign', 'Nature', 'Sky', 'Person', 'Rider', 'Car')
    PALETTE=[[i]*3 for i in range(12)]

    def __init__(self, **kwargs):
        super(DaconDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)