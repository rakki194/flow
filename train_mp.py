import torch.multiprocessing as mp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from src.trainer.train_chroma import train_chroma
# from src.trainer.double_backward_bug import train_chroma
if __name__ == "__main__":
    
    # train_chroma(0, 2, True)
    # Number of GPUs to use
    world_size = 2

    # Use spawn method for starting processes
    mp.spawn(
        train_chroma,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )