from torch.utils.data import Dataset 


class TrainDataset(Dataset):

    def __init__(self, transform):
        super().__init__()

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass

class ValDataset(Dataset):

    def __init__(self, transform):
        super().__init__()

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass