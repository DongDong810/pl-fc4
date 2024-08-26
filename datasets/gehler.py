import torch
from torch.utils import data




"""
For dataset
'datasets.py' + 'data_provider.py'



ColorChecker dataset for input images - 16 bit png files

"""




"""
Image representation (debugging)
"""
class ImageRecord:
    def __init__(self, dataset, fn, illum, mcc_coord, img, extras=None):
        self.dataset = dataset
        self.fn = fn
        self.illum = illum
        self.mcc_coord = mcc_coord
        # BRG images
        self.img = img
        self.extras = extras

    # representation (print string for object)
    def __repr__(self):
        return f'{self.dataset}, {self.fn}, {self.illum[0]}, {self.illum[1]}, {self.illum[2]}'


"""
Dataset
- use torch.util for DataLoader
"""
class Gehler(data.Dataset):
    def __init__(self, cfg, phase):
        super(Gehler, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.img_dir = cfg.data1.root
        self.data = self._load_data()
        self.transform = self._get_transform()

    # len(data)
    def __len__(self):
        return len(self.data)

    # indexing
    def __getitem__(self, idx):
        batch_dict = {}
        batch_dict['x'] = torch.randn(10)
        batch_dict['y'] = torch.randn(10)

        if self.transform is None:
            return batch_dict    
        else:
            return self.transform(batch_dict)

    def _load_data(self):
        


        return range(10000)




    def _get_transform(self):
        # Get transform
        pass


    # def load_image(self, fn):
    #     file_path = self.get_img_directory() + '/images/' + fn
    #     raw = np.array(cv2.imread(file_path, -1), dtype='float32')
    #     if fn.startswith('IMG'):
    #         # 5D3 images
    #         black_point = 129
    #     else:
    #         black_point = 1
    #     raw = np.maximum(raw - black_point, [0, 0, 0])
    #     return raw



def load_data(folds):
    records = []
    for fold in folds:
        fn = get_image_pack_fn(fold)
        print('Loading image pack', fn)
        # cached
        if fn not in load_data.data:
            with open(fn, 'rb') as f:  # 'rb' to read in binary mode
                load_data.data[fn] = pickle.load(f)
        records += load_data.data[fn]
    return records



def get_loader(cfg, phase):
    """
    Args:
        cfg: config file
        phase: ['train', 'valid', 'test']
    """

    dataset = Gehler(cfg, phase)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True, # for GPU
    )
    return loader
