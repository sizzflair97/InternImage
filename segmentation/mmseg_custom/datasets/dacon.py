from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DaconDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('Road', 'Sidwalk', 'Construction', 'Fence', 'Pole', 'Traffic Light', 'Traffic Sign', 'Nature', 'Sky', 'Person', 'Rider', 'Car'),
        palette=[])

    def __init__(self, aeg1, arg2):
        pass