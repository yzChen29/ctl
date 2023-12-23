# TCIL

This code is partial implementation of paper: Taxonomic Class Incremental Learning (accessible at https://arxiv.org/abs/2304.05547). This version only supports the cifar100 dataset which will be downloaded to /data/ folder automatically.


# Configs

Configs can be motified on /codes/base/configs/ctl2_gpu_cifar100.yaml

# How to run

Step1:

    Create a folder to store checkpoint, eg. cifar100_results
    Copy its absolute path to "save_result_path" in corresponding yaml file (find in /codes/base/configs) and motify the other parameters

Step2:

    Run the following commands to train:
    
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    cd {PATH of TCIL codes}
    pip install -r requirements.txt
    cp misc/_tensor.py {PATH of pytorch installed}/_tensor.py
    cd codes/base
    bash scripts/train_server2.sh

# Expect Output

    train/acc_details: store all the (aux)accuracy for each category for each task
    train/ckpts: store all the checkpoints at the end of each epoch
    train/logs/train.log: store all the logs during training, including loss, accuracy etc.
    

