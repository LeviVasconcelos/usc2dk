import torch.utils.data as data
import torch
import numpy as np
from datasets.humans36m import Humans36mDataset


def LoadCoupledDatasets(source, target, collate_source, collate_target, batch_size, workers):
    def collate_fn(data):
        source, target = zip(list(*data))
        return {
                "source": collate_source(source),
                "target": collate_target(target),
                }
    coupled_dataset = CoupledDataset(source, target)
    return torch.utils.data.DataLoader(coupled_dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=workers, pin_memory=False, 
                                        collate_fn=collate_fn)

class CoupledDataset(data.Dataset):
    """
    Class unifying the loading process of data and unaligned gt
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target

        self.source_order = np.array([x for x in range(0, len(self.source))])

    def shuffle(self):
        np.random.shuffle(self.source_order)

    def __getitem__(self, i):
        j = self.source_order[i % len(self.source)]
        k = i % len(self.target)
        source_item = self.source.__getitem__(j)
        target_item = self.target.__getitem__(k)
        return (source_item, target_item)

    def __len__(self):
        return len(self.target)


