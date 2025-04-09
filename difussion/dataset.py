import torch
import glob
from os.path import join
from torch.utils.data import Dataset
from tqdm import tqdm
from difussion.utils.logger import logger


class DifussionDataset(Dataset):

    valid_suffixs = ["jpg","jpeg","png"]

    def __init__(self,path:str):
        
        self.path = path
        
        self.images_pt = glob.glob(join(path,"*.pt"))

        if len(self.images_pt) == 0:
            logger.error("Dataset was not trasformed to tensors")
            raise Exception

        if len(self.images_pt) < 1000:
            logger.warning("Dataset is under 1k images")

    def __len__(self):
        return len(self.images_pt)

    def __getitem__(self,idx) -> torch.Tensor:
        
        path_image_idx = self.images_pt[idx]
        image_tensor = torch.load(path_image_idx)
        return image_tensor




