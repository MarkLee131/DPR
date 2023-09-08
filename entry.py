import os
import subprocess
from sys import argv
import torch
'''
cmd is:
python -m torch.distributed.launch --nproc_per_node=4
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir=./output_test
'''
### we need to run this command in the root directory of DPR

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)

### run the command

n_process_per_node = str(4)




python_file = 'train_dense_encoder.py'
train_datasets = ['nq_train']
dev_datasets = ['nq_dev']
output_dir = './output_test'



os.chdir('/mnt/local/Baselines_Bugs/DPR')

os.environ['WORLD_SIZE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["NCCL_DEBUG"] = "INFO"
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:65534', rank=0, world_size=1)

cmd = f'python -m torch.distributed.launch --nproc_per_node={n_process_per_node} {python_file} train=biencoder_nq train_datasets={train_datasets} dev_datasets={dev_datasets} output_dir={output_dir}'

res = subprocess.run(cmd, shell=True, check=True)

print(res)



'''
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
'''