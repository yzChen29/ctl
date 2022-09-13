import pynvml
import os
import numpy as np


def check_shm_usage():
    process = os.popen('df -h')
    preprocessed = process.read()
    process.close()

    shm_info = preprocessed.split('\n')[1:-1]
    for ind in range(len(shm_info)):
        device_info_ind = [i for i in shm_info[ind].split(' ') if i != '']
        if device_info_ind[-1] == '/dev/shm':
            print(f'shm Total {device_info_ind[3]}, Use {device_info_ind[2]} —— {device_info_ind[4]}')
            break

check_shm_usage()
print('\n')

def check_cpu_info(cpu_count=10):
    process = os.popen('top -b -n 1')
    preprocessed = process.read()
    process.close()
    cpu_info = preprocessed.split('\n')[7:7+cpu_count]
    for ind in range(len(cpu_info)):
        device_info_ind = [i for i in cpu_info[ind].split(' ') if i != '']
        print(f'CPU PID {device_info_ind[0]}, Usage {device_info_ind[8]}%, Memory {device_info_ind[9]}%')

check_cpu_info()
print('\n')


def nvidia_info():
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 4,
        "gpus": []
    }
    try:
        pynvml.nvmlInit()
        nvidia_dict["nvidia_version"] = pynvml.nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = pynvml.nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": pynvml.nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{pynvml.nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": pynvml.nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except pynvml.NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            npynvml.vmlShutdown()
        except:
            pass
    return nvidia_dict


def check_gpu_mem_usedRate():
    gpus_info = nvidia_info()['gpus']

    for i in range(len(gpus_info)):
        gpu_info_i = gpus_info[i]
        print(f"GPU {i}, Name {gpu_info_i['gpu_name']}, Usage {np.round(gpu_info_i['used'] / gpu_info_i['total'] * 100, 3)}%")

check_gpu_mem_usedRate()
