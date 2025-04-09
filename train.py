import hydra
from torch.utils.data import DataLoader
from difussion.utils.utils import transform_images_to_torch,clear_data
from difussion.dataset import DifussionDataset
from difussion.utils.utils import logger

@hydra.main(version_base="1.2", config_path="cfg", config_name="train")
def main(cfg):
    
    logger.info(cfg)

    clear_data(path=cfg.train.path_dataset)

    transform_images_to_torch(path=cfg.train.path_dataset,
                              override=True)

    dataset = DifussionDataset(path=cfg.train.path_dataset)

    dataloader = DataLoader(dataset,
                            batch_size=cfg.train.batch_size)
    
    for batch in dataloader:
        print(batch.shape)
    
if __name__ == '__main__':
    main()
    
    