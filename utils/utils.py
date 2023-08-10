import  torch
import  random
import os
import  torch.nn            as      nn
import  numpy               as      np


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)