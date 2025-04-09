import os
import glob
import torch
from torchvision.transforms import Resize
import numpy as np
from os.path import join
from tqdm import tqdm
from PIL import Image
from difussion.utils.logger import logger


def transform_images_to_torch(path:str,
                              override=False,
                              img_size:int = 256):


    logger.info("Creating dataset...")
    
    valid_suffixs = ["jpg","jpeg","png"]

    images_path = []
    errors = 0

    for suffix in valid_suffixs:

        images_path += glob.glob(join(path,f"*.{suffix}"))
    
    for image_path in tqdm(images_path,total=len(images_path)):

        try:
            name_image = image_path.split('/')[-1]
            image_tensor_path = join(path,f'{name_image}.pt')
            if os.path.exists(image_tensor_path) and not override:
                continue
            else:
                image_pil = Image.open(image_path)
                image_array = np.array(image_pil)
                image_tensor = torch.tensor(image_array,dtype=torch.float32).permute(2,0,1)
                c,h,w = image_tensor.shape
                if c != 3:
                    raise Exception
                image_tensor = Resize(size=(img_size,img_size))(image_tensor)
                torch.save(image_tensor,image_tensor_path)
        except:
            errors += 1
            logger.warning(f"Image {image_path} not transformed")

    logger.success(f"Dataset created:\nTotal images: {len(images_path)-errors}\nErrors:{errors}")

def clear_data(path:str):

    images_pt_files = glob.glob(join(path,"*.pt"))

    for file in images_pt_files:
        os.remove(file)