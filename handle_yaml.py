import pandas as pd
import os

# a ='''ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance cifar100(sp0.3) fs [512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance cifar100(sp0.3) fs [512, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance cifar100(sp0.3) fs [512, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance cifar100(sp0.3) fs [512, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full cifar100(sp0.3) fs [512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full cifar100(sp0.3) fs [512, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full cifar100(sp0.3) fs [512, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full cifar100(sp0.3) fs [512, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [512, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [512, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [512, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [256, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect cifar100(sp0.3) fs [256, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance random_order cifar100(sp0.3) fs [512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full random_order cifar100(sp0.3) fs [512, 512]
#
#
#
#
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance imagenet100 fs [512, 512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance imagenet100 fs [512, 256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance imagenet100 fs [512, 128, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance imagenet100 fs [512, 256, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full imagenet100 fs [512, 512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full imagenet100 fs [512, 256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full imagenet100 fs [512, 128, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full imagenet100 fs [512, 256, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect imagenet100 fs [512, 512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect imagenet100 fs [512, 256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect imagenet100 fs [512, 128, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# connect imagenet100 fs [512, 256, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect only_ance random_order imagenet100 fs [512, 512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# wo connect full random_order imagenet100 fs [512, 512, 512]
#
#
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [512, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [512, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [512, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [256, 256]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline cifar100(sp0.3) fs [256, 64]
#
#
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline imagenet100 fs [512, 512, 512]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline imagenet100 fs [512, 256, 128]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline imagenet100 fs [512, 128, 64]
#
# ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t
# der baseline imagenet100 fs [512, 256, 64]
#
# '''
#
# res  =  []
# flag=0
# for i in a.split('\n'):
#     if flag == 1:
#         res.append(i)
#         flag=0
#     if i == 'ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-6fj2t':
#         flag=1
#
# dataset_list = []
# model_type_list = []
# fs_list = []
#
# for i in res:
#     if 'cifar100(sp0.3)' in i:
#         dataset='cifar100(sp0.3)'
#         model_type = i.split(' cifar100(sp0.3) ')[0]
#         model_type = model_type.replace(' ', '_')
#     else:
#         dataset = 'imagenet100'
#         model_type = i.split(' imagenet100 ')[0]
#         model_type = model_type.replace(' ', '_')
#     fs_list_tmp = i.split(' fs ')[1][1:-1].split(', ')
#     fs = tuple(int(j) for j in fs_list_tmp)
#     dataset_list.append(dataset)
#     model_type_list.append(model_type)
#     fs_list.append(fs)
#
#
# res_dict = {'job':list(range(47, 47+len(res))), 'task':res, 'dataset': dataset_list, 'model_type': model_type_list, 'fs': fs_list}
# for i in res_dict:
#     print(i, len(res_dict[i]))
# df = pd.DataFrame(res_dict)
# df.to_csv('/Users/chenyuzhao/Desktop/UCSD项目/server/job/job_info.csv', index=False)



def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def gen_yaml_setting(cpu_num, job_ind, model_name, train_server, save_path):
    info = f'''
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: ctl-imagenet-{cpu_num}cpu-1gpu-64mem-pvc-datasets2-job-{job_ind}
    
    spec:
      template:
        spec:
          volumes:
          - name: datasets
            persistentVolumeClaim:
              claimName: datasets-2
          
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                    - NVIDIA-GeForce-RTX-3090
                      #- NVIDIA-GeForce-RTX-2080-Ti
                    - NVIDIA-TITAN-RTX
                    - NVIDIA-RTX-A5000
                    - NVIDIA-RTX-A6000
                    - NVIDIA-A40
    
                  - key: topology.kubernetes.io/region
                    operator: In
                    values:
                    - us-west
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                      - test
                      #- fiona8-01.sdsu.edu
                      #- k8s-gpu-3.ucsc.edu
                  - key: nvidia.com/cuda.driver.major
                    operator: Gt
                    values:
                    - "509"
                  - key: nvidia.com/cuda.driver.minor
                    operator: Gt
                    values:
                    - "46"
                    
          containers:
          - name: mypod
            image: centos/python-38-centos7
            volumeMounts:
            - name: datasets
              mountPath: /datasets
            - name: shm
              mountPath: /dev/shm
            # copy setting        ctl_name / yaml / main (vscode)
            # GPU / CPU number:   4/8(resources)
            # ctl name:           codes/ctl_imagenet100_connect_fs512_128_64 (args)
            # sh file name:       train_server (args)
            
            resources:
               limits:
                 memory: 64Gi
                 cpu: {cpu_num}
                 nvidia.com/gpu: 1
               requests:
                 memory: 64Gi
                 cpu: {cpu_num}
                 nvidia.com/gpu: 1
            command: ["/bin/sh", "-c"]
    
            args:
              - >
                cp -r /datasets/codes/{model_name}/ctl ./ &&
                pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html &&
                pip install -r ctl/requirements.txt &&
                cp ~/ctl/misc/_tensor.py /opt/app-root/lib64/python3.8/site-packages/torch/_tensor.py &&
                cd ~/ctl/codes/base &&
                bash scripts/{train_server}.sh
    
          volumes:
          - name: datasets
            persistentVolumeClaim:
                claimName: datasets-2
    
          - name: shm
            emptyDir:
              medium: Memory
    
          restartPolicy: Never
      backoffLimit: 4
    '''

    res = '\n'.join([i[4:] for i in info.split('\n')][1:])

    with open(f'{save_path}/job_{job_ind}.yaml', 'w') as f:
        f.write(res)


def gen_config(model_name, fs_list, job_ind, save_path):
    use_joint_ce_loss = True
    feature_mode = 'add_zero_only_ancestor_fea'
    use_connection = True
    if 'wo_connect_full' in model_name:
        use_joint_ce_loss = False
        feature_mode = 'full'

    if 'wo_connect' in model_name:
        use_connection = False
    # model_name =  "ctl_imagenet100_connect_fs512_128_64_more_data_Mar3"
    # use_joint_ce_loss = True
    # feature_mode = 'add_zero_only_ancestor_fea'
    # use_connection = True
    # fs_list = [512, 128, 64]

    info_imagenet =f'''
# for imagenet100

exp:
  name: "{model_name}"
  load_model_name: '/datasets/imagenet100_results/{model_name}'
  debug: False

# important cfg

# cfg_check_1
debug: False
memory_enable: True

use_joint_ce_loss: {use_joint_ce_loss}
feature_mode: '{feature_mode}' # ['full', 'add_zero_only_ancestor_fea',  'add_zero_use_all_prev']  # 新任务选用哪些feature

batch_size: 256        # 256 or 64
num_workers: 8
save_ckpt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

auto_retrain: True
bn_reset_running: False
bn_no_tracking: False
zero_init_residual: True

validation: 0  # Validation split (0. <= x <= 1.)

full_connect: False
use_connection: {use_connection}  # True or False

connect_fs: {fs_list}        # 512 or 128
sample_rate_c1: 0.3   # 0.3 or 0.001
sample_rate_c2: 0.21  # 0.21 or 0.001
epochs: 220           # 220 or 2
decouple:
  enable: True
  epochs: 30           # 30 or 2
  fullset: False
  lr: 0.1
  scheduling:
    - 15
  lr_decay: 0.1
  weight_decay: 0.0005
  temperature: 1.0


# cfg_check_2
# load_prev_model eg. has ckpts/ (decouple_)step13 then set retrain_from_task: 14
# use_joint_ce_loss: True
# feature_mode: 'add_zero_only_ancestor_fea' # ['full', 'add_zero_only_ancestor_fea',  'add_zero_use_all_prev']  # 新任务选用哪些feature
retrain_from_task: 0  # 0 or n
save_mem: True
load_mem: False       # False or True
show_noutput_detail: True   # True or False

# cfg_check_3
# change seed & dataset
seed: 500              # 500 or 600 or 1993
dataset: "imagenet100" # 'imagenet100' or 'cifar100'
trial: 3               # 3 or 2
acc_k: 0               # record top k acc   =0 then no res

# check datasets.py trial2 / trial3 BFS / DFS


# check cgpu_using
show_detail: False    # show time consumsing or not
check_cgpu_info: False
check_cgpu_batch_period: 30

# cfg_check_4
# change batch_size then change the lr in  decouple
lr: 0.1
# batch_size: 256        # 256 or 64
# num_workers: 4

save_result_path: '/datasets/imagenet100_results/' # '/datasets/imagenet100_results/' or ''

taxonomy: 'rtc' # rtc for Real Taxonomic Classifier
# save_ckpt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
save_before_decouple: False


# lr_decay
scheduling:
  - 60
  - 120
  - 160
  - 180

is_distributed: False


# Model Cfg
model: "incmodel"
model_cls: "hiernet"
model_pivot:
  arch: pivot
  dropout_rate: 0.0
convnet: 'resnet18'  # modified_resnet32, resnet18
#taxonomy: 'rtc' # rtc for Real Taxonomic Classifier
train_head: 'softmax'
infer_head: 'softmax'
channel: 64
use_bias: False
last_relu: False

der: True
use_aux_cls: True
aux_n+1: True
#use_joint_ce_loss: True
distillation: "none"
temperature: 2

reuse_oldfc: True
weight_normalization: False
val_per_n_epoch: -1 # Validation Per N epoch. -1 means the function is off.
#save_ckpt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
#save_result_path: '/datasets/imagenet100_results/' # '/datasets/imagenet100_results/' or ''
#check_cgpu_info: True
#check_cgpu_batch_period: 30
#save_mem: True
#load_mem: False

# Optimization; Training related
task_max: 10
lr_min: 0.00005
#lr: 0.1
#num_workers: 4
weight_decay: 0.0005
dynamic_weight_decay: False
scheduler: 'multistep'
#scheduling:
#  - 100
#  - 120
lr_decay: 0.1
optimizer: "sgd"
#epochs: 2
resampling: False
warmup: False
warmup_epochs: 10
#sample_rate: 0.001
#retrain_from_task: 0
#acc_k: 0               # record top k acc   =0 then no res

train_save_option:
  acc_details: True
  acc_aux_details: True
  preds_details: True
  preds_aux_details: True

postprocessor:
  enable: False
  type: 'bic'  #'bic', 'wa'
  epochs: 1
  batch_size: 128
  lr: 0.1
  scheduling:
    - 60
    - 90
    - 120
  lr_decay_factor: 0.1
  weight_decay: 0.0005

pretrain:
  epochs: 200
  lr: 0.1
  scheduling:
    - 60
    - 120
    - 160
  lr_decay: 0.1
  weight_decay: 0.0005


# Dataset Cfg
#dataset: "imagenet100"  # 'imagenet100', 'cifar100'
#trial: 3
increment: 10
#batch_size: 128

# validation: 0.1  # Validation split (0. <= x <= 1.)
random_classes: False  # Randomize classes order of increment
start_class: 0  # number of tasks for the first step, start from 0.
start_task: 0
max_task:  # Cap the number of task
#is_distributed: False

# Memory
# memory_enable: False
coreset_strategy: "iCaRL"  # iCaRL, random
mem_size_mode: "uniform_fixed_total_mem"  # uniform_fixed_per_cls, uniform_fixed_total_mem
memory_size: 2000  # Max number of storable examplars
fixed_memory_per_cls: 20  # the fixed number of exemplars per cls

# Misc
device_auto_detect: True  # If True, use GPU whenever possible
device: 0 # GPU index to use, for cpu use -1
#seed: 500
overwrite_prevention: False

    '''

    info_cifar = f'''
# for cifar100

# check name / load_model_name / connect_fs
exp:
  name: {model_name}
  load_model_name: '/datasets/cifar100_results/{model_name}'
  debug: False

# important cfg

# cfg_check_1
debug: False
memory_enable: True

use_joint_ce_loss: {use_joint_ce_loss}
feature_mode: '{feature_mode}' # ['full', 'add_zero_only_ancestor_fea',  'add_zero_use_all_prev']  # 新任务选用哪些feature

auto_retrain: True
bn_reset_running: False
bn_no_tracking: False
zero_init_residual: True

validation: 0  # Validation split (0. <= x <= 1.)

full_connect: False
use_connection: {use_connection}  # True or False

connect_fs: {fs_list}        # 512 or 128
sample_rate_c1: 0.21   # 0.3 or 0.001
sample_rate_c2: 0.3  # 0.21 or 0.001
epochs: 170           # 170 or 2
decouple:
  enable: True
  epochs: 50          # 50 or 2
  fullset: False
  lr: 0.05            # 0.1 or 0.05
  scheduling:
    - 15
    - 30
  lr_decay: 0.1
  weight_decay: 0.0005
  temperature: 5.0


# cfg_check_2
# load_prev_model eg. has ckpts/ (decouple_)step13 then set retrain_from_task: 14
# use_joint_ce_loss: True
# feature_mode: 'add_zero_only_ancestor_fea' # ['full', 'add_zero_only_ancestor_fea',  'add_zero_use_all_prev']  # 新任务选用哪些feature
retrain_from_task: 0  # 0 or n
save_mem: True
load_mem: False       # False or True
show_noutput_detail: True   # True or False

# cfg_check_3
# change seed & dataset
seed: 500              # 500 or 600 or 1993
dataset: "cifar100" # 'imagenet100' or 'cifar100'
trial: 5               # 3 or 2
acc_k: 0               # record top k acc   =0 then no res

# check datasets.py trial2 / trial3 BFS / DFS


# check cgpu_using
show_detail: False    # show time consumsing or not
check_cgpu_info: False
check_cgpu_batch_period: 30

# cfg_check_4
# change batch_size then change the lr in  decouple
lr: 0.1
batch_size: 128        # 256 or 64
num_workers: 4

save_result_path: '/datasets/cifar100_results/' # '/datasets/imagenet100_results/' or ''

taxonomy: 'rtc' # rtc for Real Taxonomic Classifier
save_ckpt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
save_before_decouple: False


# lr_decay
scheduling:
  - 100
  - 120

is_distributed: False


# Model Cfg
model: "incmodel"
model_cls: "hiernet"
model_pivot:
  arch: pivot
  dropout_rate: 0.0
convnet: 'resnet18'  # modified_resnet32, resnet18
#taxonomy: 'rtc' # rtc for Real Taxonomic Classifier
train_head: 'softmax'
infer_head: 'softmax'
channel: 64
use_bias: False
last_relu: False

der: True
use_aux_cls: True
aux_n+1: True
#use_joint_ce_loss: True
distillation: "none"
temperature: 2

reuse_oldfc: True
weight_normalization: False
val_per_n_epoch: -1 # Validation Per N epoch. -1 means the function is off.
#save_ckpt: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
#save_result_path: '/datasets/imagenet100_results/' # '/datasets/imagenet100_results/' or ''
#check_cgpu_info: True
#check_cgpu_batch_period: 30
#save_mem: True
#load_mem: False

# Optimization; Training related
task_max: 10
lr_min: 0.00005
#lr: 0.1
#num_workers: 4
weight_decay: 0.0005
dynamic_weight_decay: False
scheduler: 'multistep'
#scheduling:
#  - 100
#  - 120
lr_decay: 0.1
optimizer: "sgd"
#epochs: 2
resampling: False
warmup: False
warmup_epochs: 10
#sample_rate: 0.001
#retrain_from_task: 0
#acc_k: 0               # record top k acc   =0 then no res

train_save_option:
  acc_details: True
  acc_aux_details: True
  preds_details: True
  preds_aux_details: True

postprocessor:
  enable: False
  type: 'bic'  #'bic', 'wa'
  epochs: 1
  batch_size: 128
  lr: 0.1
  scheduling:
    - 60
    - 90
    - 120
  lr_decay_factor: 0.1
  weight_decay: 0.0005

pretrain:
  epochs: 200
  lr: 0.1
  scheduling:
    - 60
    - 120
    - 160
  lr_decay: 0.1
  weight_decay: 0.0005


# Dataset Cfg
#dataset: "imagenet100"  # 'imagenet100', 'cifar100'
#trial: 3
increment: 10
#batch_size: 128

# validation: 0.1  # Validation split (0. <= x <= 1.)
random_classes: False  # Randomize classes order of increment
start_class: 0  # number of tasks for the first step, start from 0.
start_task: 0
max_task:  # Cap the number of task
#is_distributed: False

# Memory
# memory_enable: False
coreset_strategy: "iCaRL"  # iCaRL, random
mem_size_mode: "uniform_fixed_total_mem"  # uniform_fixed_per_cls, uniform_fixed_total_mem
memory_size: 2000  # Max number of storable examplars
fixed_memory_per_cls: 20  # the fixed number of exemplars per cls

# Misc
device_auto_detect: True  # If True, use GPU whenever possible
device: 0 # GPU index to use, for cpu use -1
#seed: 500
overwrite_prevention: False


'''

    make_dir(f'{save_path}/{model_name}_ind{job_ind}')
    if 'imagenet100' in model_name:
        info = info_imagenet
        file_name = 'ctl2_gpu_imagenet100.yaml'
    else:
        info = info_cifar
        file_name = 'ctl2_gpu_cifar100.yaml'
    with open(f'{save_path}/{model_name}_ind{job_ind}/{file_name}', 'w') as f:
        f.write(info)
    # return file_name

def gen_server_command(model_name, job_ind):
    if 'imagenet100' in model_name:
        dataset = 'imagenet'
    else:
        dataset = 'cifar'
    print(f'mkdir /datasets/codes/{model_name}')
    print(f'cp -r ~/ctl /datasets/codes/{model_name}')
    if not 'random_order' in model_name:
        print(f'cp ~/config_folder/{model_name}_ind{job_ind}/ctl2_gpu_{dataset}100.yaml /datasets/codes/{model_name}/ctl/codes/base/configs')
        print(f'cp ~/addition_update/non_baseline_{dataset}_main/main.py /datasets/codes/{model_name}/ctl/codes/base')
    print('\n\n')

def gen_job_command(job_ind):
    # print(f'kubectl delete -f /Users/chenyuzhao/Desktop/UCSD项目/server/job/gen_job_info/job_{job_ind}.yaml')

    print(f'kubectl create -f /Users/chenyuzhao/Desktop/UCSD项目/server/job/gen_job_info/job_{job_ind}.yaml')


pod_mesg = '''cogrob-7bc56f8466-jl8rf                                           1/1     Running                  0          7d21h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-15-pq6kz           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-16-jkcrm           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-17-t47mx           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-18-v64vf           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-19-wbxt4           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-20-gsfv5           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-21-6hhv7           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-22-bgrxp           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-23-dtlhc           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-24-lqchx           0/1     Completed                0          3d7h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-25-5tsbj           0/1     Completed                0          2d20h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-26-bgkpj           0/1     Completed                0          2d20h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-27-b7755           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-28-mwpz4           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-29-7rb4s           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-30-n4l8r           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-32-j6jmx           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-33-4z7h8           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-34-958kb           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-35-x76bk           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-36-nkwwr           0/1     Completed                0          2d3h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-43-fjwzm           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-44-knd98           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-45-mfvdl           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-46-7n2qv           0/1     Completed                0          43h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-47-nrk7h           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-48-ttxfv           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-49-gs9q8           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-50-qsxdz           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-51-5j5kk           0/1     Completed                0          14h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-52-tw6gn           0/1     Completed                0          14h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-53-wnsnv           0/1     Completed                0          14h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-54-jz8mh           0/1     Completed                0          14h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-55-jkdhz           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-56-zxsq4           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-57-s5xxl           0/1     Completed                0          22h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-57-sxf2m           0/1     ContainerStatusUnknown   1          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-58-l9wjf           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-59-bmzsb           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-60-9nqrx           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-61-ch5s8           0/1     Completed                0          29h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-62-zvdmw           0/1     Completed                0          27h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-63-mt45q           0/1     Completed                0          14h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-78-2k7m2           0/1     Completed                0          22h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-78-s5jwv           0/1     Error                    0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-79-6qz46           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-80-vznb5           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-81-7684k           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-82-6dcwk           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-83-vcxw9           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-job-84-nhs56           0/1     Completed                0          25h
ctl-imagenet-4cpu-1gpu24g-32mem-pvc-datasets2-1-577bf649f-8dwcw   1/1     Running                  0          7m31s
ctl-imagenet-4cpu-1gpu24g-32mem-pvc-datasets2-11-65b58697688qcj   1/1     Running                  0          12h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-1-4cwpd            0/1     Completed                0          4d8h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-11-rczzb           0/1     Completed                0          3d7h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-12-6tpv5           0/1     Completed                0          3d7h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-13-pkhfp           0/1     Completed                0          3d7h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-14-69g4q           0/1     Error                    0          3d7h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-14-mww8k           1/1     Running                  0          13h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-3-sw7jz            0/1     Completed                0          4d9h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-37-7dp5l           0/1     Completed                0          2d3h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-38-bjc4f           0/1     Completed                0          43h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-39-rzmq7           0/1     Completed                0          43h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-40-d2h79           0/1     Completed                0          43h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-41-bpllm           0/1     Completed                0          33h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-42-trp5h           0/1     Completed                0          43h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-64-26wmv           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-65-t5fbl           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-66-lhhbl           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-67-hpncw           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-68-cwhrh           1/1     Running                  0          14h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-69-khx9w           1/1     Running                  0          14h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-70-d78q6           1/1     Running                  0          7h19m
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-70-g2hb6           0/1     Error                    0          14h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-71-sm456           1/1     Running                  0          14h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-72-hx4lt           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-73-8r8bd           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-74-qctql           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-75-drxzq           1/1     Running                  0          29h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-76-pfrx6           1/1     Running                  0          27h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-77-7dmj2           1/1     Running                  0          14h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-85-vxkdf           1/1     Running                  0          25h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-86-bxw7d           1/1     Running                  0          25h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-87-mc2h4           1/1     Running                  0          25h
ctl-imagenet-8cpu-1gpu-64mem-pvc-datasets2-job-88-qwfpq           1/1     Running                  0          25h
ctl-imagenet-con2-668f9c745d-2n8jk                                1/1     Running                  0          2d23h
debug                                                             0/1     Error                    0          16h
dne-job-dytox-12heads-imagenet100-order2-sub-9kz48                1/1     Running                  0          34h
dne-job-dytox-12heads-imagenet100-order3-j6lmq                    1/1     Running                  0          34h
dne-job-dytox-16heads-order2-x86zg                                0/1     Completed                0          22h
egovlp-dep-85fc7684fd-59ghg                                       1/1     Running                  0          3d21h
pycil-job-der-imgnet100-b64-order2-wpnrx                          1/1     Running                  0          38h
pycil-job-der-imgnet100-b64-order3-4cplk                          1/1     Running                  0          11h
pycil-job-foster-b96-order2-nsdf4                                 0/1     Completed                0          22h
pycil-job-foster-imgnet100-b64-order3-txw4g                       0/1     Completed                0          33h
pycil-job-icarl-imgnet100-b64-order3-bj6rr                        1/1     Running                  0          12h
pycil-job-podnet-imgnet100-b64-order3-rw2tw                       0/1     Completed                0          33h'''

def analysis_pod(pod_mesg, start_index):
    job_list = []
    total_list = list(range(47, 90))

    for i in pod_mesg.split('\n'):
        if 'cpu-1gpu-64mem-pvc-datasets2-job-' in i:
            pod_index = int(i.split('-job-')[1].split('-')[0])
            if pod_index  in total_list:
                total_list.remove(pod_index)
            if pod_index >= start_index:
                job_list.append(pod_index)
    return job_list, total_list

def show_pod_log(pod_mesg, index, ind2name, start_index):
    complete_ind_list = []
    complete_name_list = []
    for i in pod_mesg.split('\n'):
        if 'cpu-1gpu-64mem-pvc-datasets2-job-' in i:
            pod_index = int(i.split('-job-')[1].split('-')[0])
            if pod_index >= start_index and 'Completed' in i:
                complete_ind_list.append(pod_index)
                complete_name_list.append(ind2name[pod_index])
            if pod_index == index:
                print(f'kubectl logs {i}')

    return complete_ind_list, complete_name_list

def addition_server_command(model_name):
    print(f'cp ~/ctl/inclearn/models/incmodel.py /datasets/codes/{model_name}/ctl/inclearn/models/incmodel.py')
    print(f'cp ~/ctl/inclearn/convnet/utils.py /datasets/codes/{model_name}/ctl/inclearn/convnet/utils.py')
    print('\n\n')

date = 'Mar3'
df = pd.read_csv('/Users/chenyuzhao/Desktop/UCSD项目/server/job/job_info.csv')
yaml_save_path = '/Users/chenyuzhao/Desktop/UCSD项目/server/job/gen_job_info'
config_save_path = '/Users/chenyuzhao/Desktop/UCSD项目/server/job/config_folder'

count = 0

pod_list, total_list = analysis_pod(pod_mesg, 47)
print(total_list)
ind2name = {}
for i in range(df.shape[0]):

# for i in range(1):

    addition_info = ''
    job_ind = df.iloc[i, 0]
    dataset = df.iloc[i, 2]
    model_type = df.iloc[i, 3]
    fs = df.iloc[i, 4]
    fs_list_tmp = list(fs[1:-1].split(', '))
    fs_list = [int(i) for i in fs_list_tmp]

    if dataset == 'cifar100(sp0.3)':
        dataset_name = 'cifar100'
        addition_info = 'sp03_more_data'
    else:
        dataset_name = 'imagenet100'
        addition_info = 'more_data'
    if len(fs_list) == 2:
        fs_name  = f'fs{fs_list[0]}_{fs_list[1]}'
    else:
        fs_name = f'fs{fs_list[0]}_{fs_list[1]}_{fs_list[2]}'
    model_name = f'ctl_{dataset_name}_{model_type}_{fs_name}_{addition_info}_{date}'
    ind2name[job_ind] = model_name

    if 'cifar' in dataset:
        cpu_num = 4
        train_server = 'train_server2'
    else:
        cpu_num = 8
        train_server = 'train_server'
    # gen_yaml_setting(cpu_num, job_ind, model_name, train_server, save_path=yaml_save_path)

    # if 'der_baseline' not in model_name and 'random_order' not in model_name:
        # count += 1
        # gen_config(model_name, fs_list, job_ind=job_ind, save_path = config_save_path)
        # gen_server_command(model_name, job_ind)
        # gen_job_command(job_ind)

        # if job_ind not in pod_list:
        #     gen_job_command(job_ind)

    # if 'random_order' in model_name:
        # gen_server_command(model_name, job_ind)

    # if 'full' in model_name:
    #     print(model_name)

        # No No No
        # addition_server_command(model_name)
        # gen_job_command(job_ind)




complete_ind_list, complete_name_list = show_pod_log(pod_mesg, 78, ind2name, 47)
for i in  complete_name_list:
    if 'imagenet' in i:
        print(i)
print(complete_ind_list)
print(complete_name_list)
print(list(zip(complete_ind_list, complete_name_list)))

# print(count)


# randmom_order = [[276, 337, 379, 269, 279], [138, 565, 20, 471, 880], [728, 595, 85, 10, 519], [334, 725, 274, 292, 145], [344, 492, 352, 99, 17], [876, 371, 282, 772, 19], [695, 146, 480, 139, 414], [739, 513, 771, 694, 676], [131, 380, 769, 847, 11], [844, 12, 16, 748, 18], [636, 670, 797, 709, 14], [84, 81, 135, 336, 637], [464, 584, 129, 333, 100], [268, 368, 746, 398, 666], [491, 829, 338, 583, 345], [378, 479, 283, 86, 755], [340, 618, 82, 776, 561], [612, 875, 438, 428, 13], [136, 527, 291, 293, 351], [594, 837, 866, 683, 757]]
# BFS_list = [['hyena', 'beaver', 'howler_monkey', 'timber_wolf', 'arctic_fox'], ['bustard', 'freight_car', 'water_ouzel', 'cannon', 'unicycle'], ['plastic_bag', 'harvester', 'quail', 'brambling', 'crate'], ['porcupine', 'pitcher', 'dhole', 'tiger', 'king_penguin'], ['hippopotamus', 'chest', 'impala', 'goose', 'jay'], ['tub', 'patas', 'tiger_cat', 'safety_pin', 'chickadee'], ['padlock', 'albatross', 'cash_machine', 'ruddy_turnstone', 'backpack'], ['potters_wheel', 'cornet', 'safe', 'paddlewheel', 'muzzle'], ['little_blue_heron', 'titi', 'rule', 'tank', 'goldfinch'], ['switch', 'house_finch', 'bulbul', 'purse', 'magpie'], ['mailbag', 'motor_scooter', 'sleeping_bag', 'pencil_box', 'indigo_bunting'], ['peacock', 'ptarmigan', 'limpkin', 'marmot', 'mailbox'], ['buckle', 'hair_slide', 'spoonbill', 'hamster', 'black_swan'], ['mexican_hairless', 'gibbon', 'puck', 'abacus', 'mortar'], ['chain_saw', 'streetcar', 'guinea_pig', 'guillotine', 'ox'], ['capuchin', 'car_wheel', 'persian_cat', 'partridge', 'radio_telescope'], ['zebra', 'ladle', 'ruffed_grouse', 'sax', 'forklift'], ['jinrikisha', 'trombone', 'beaker', 'barrow', 'junco'], ['european_gallinule', 'desktop_computer', 'lion', 'cheetah', 'hartebeest'], ['harp', 'sunglasses', 'tractor', 'oboe', 'recreational_vehicle']]
#
# data_name_hier_dict_100_trial3 = {'mammal': {'ungulate': {'hyena': {}, 'beaver': {}, 'howler_monkey': {}, 'timber_wolf': {}, 'arctic_fox': {}}, 'rodent': {'bustard': {}, 'freight_car': {}, 'water_ouzel': {}, 'cannon': {}, 'unicycle': {}}, 'primate': {'plastic_bag': {}, 'harvester': {}, 'quail': {}, 'brambling': {}, 'crate': {}}, 'feline': {'porcupine': {}, 'pitcher': {}, 'dhole': {}, 'tiger': {}, 'king_penguin': {}}, 'canine': {'hippopotamus': {}, 'chest': {}, 'impala': {}, 'goose': {}, 'jay': {}}}, 'bird': {'game_bird': {'tub': {}, 'patas': {}, 'tiger_cat': {}, 'safety_pin': {}, 'chickadee': {}}, 'finch': {'padlock': {}, 'albatross': {}, 'cash_machine': {}, 'ruddy_turnstone': {}, 'backpack': {}}, 'wading_bird': {'potters_wheel': {}, 'cornet': {}, 'safe': {}, 'paddlewheel': {}, 'muzzle': {}}, 'other_oscine': {'little_blue_heron': {}, 'titi': {}, 'rule': {}, 'tank': {}, 'goldfinch': {}}, 'other_aquatic_bird': {'switch': {}, 'house_finch': {}, 'bulbul': {}, 'purse': {}, 'magpie': {}}}, 'device': {'instrument': {'mailbag': {}, 'motor_scooter': {}, 'sleeping_bag': {}, 'pencil_box': {}, 'indigo_bunting': {}}, 'restraint': {'peacock': {}, 'ptarmigan': {}, 'limpkin': {}, 'marmot': {}, 'mailbox': {}}, 'mechanism': {'buckle': {}, 'hair_slide': {}, 'spoonbill': {}, 'hamster': {}, 'black_swan': {}}, 'musical_instrument': {'mexican_hairless': {}, 'gibbon': {}, 'puck': {}, 'abacus': {}, 'mortar': {}}, 'machine': {'chain_saw': {}, 'streetcar': {}, 'guinea_pig': {}, 'guillotine': {}, 'ox': {}}}, 'container': {'vessel': {'capuchin': {}, 'car_wheel': {}, 'persian_cat': {}, 'partridge': {}, 'radio_telescope': {}}, 'box': {'zebra': {}, 'ladle': {}, 'ruffed_grouse': {}, 'sax': {}, 'forklift': {}}, 'bag': {'jinrikisha': {}, 'trombone': {}, 'beaker': {}, 'barrow': {}, 'junco': {}}, 'self-propelled_vehicle': {'european_gallinule': {}, 'desktop_computer': {}, 'lion': {}, 'cheetah': {}, 'hartebeest': {}}, 'other_wheeled_vehicle': {'harp': {}, 'sunglasses': {}, 'tractor': {}, 'oboe': {}, 'recreational_vehicle': {}}}}

