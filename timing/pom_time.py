# -*- coding: utf-8 -*-

# pyorimap/timing/time.py

"""
Measure execution time of elementary operations.

This module contains the following functions:

- `time_qmult()` - generate ...
"""
import os
import subprocess

import logging
import numpy as np
import cupy as cp
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt

import quaternions_np as q4np
import quaternions_cp as q4cp
import quaternions_numba_cpu as q4nCPU
import quaternions_numba_gpu as q4nGPU

from cupyx.profiler import benchmark
from cpuinfo import get_cpu_info

logging.basicConfig(level=logging.INFO)

DTYPEf = np.float32
DTYPEi = np.int32

uname = os.uname()
system = uname[0]
thehost = uname[1]
logging.info("Host, system: {}, {}".format(thehost, system))

cpu_info = get_cpu_info()
logging.info("CPU model, arch: {}, {}".format(cpu_info['brand_raw'], cpu_info['arch']))

graphicscard = subprocess.check_output("nvidia-smi -L", shell=True)
graphicscard = graphicscard.decode("ascii").split('(')[0].rstrip()

device = cp.cuda.Device(0)
memGPUleft = device.mem_info[0]
memGPUtot = device.mem_info[1]

logging.info("GPU device, memory (Mo): {}, {}".format(graphicscard, memGPUtot/1024**2))

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def time_qmult(nmax=64*1024**2):
    """
    Timing elementary quaternion multiplication.
    """
    start = 1024
    stop = 64*1024*1024
    size = np.array([1024*2**i for i in range(20) if 1024*2**i <= stop], dtype=np.int32)[::-1]

    qa_cpu = q4np.q4_random(size[0])
    qb_cpu = qa_cpu[::-1,:]

    qa_gpu = cp.asarray(qa_cpu, dtype=DTYPEf)
    qb_gpu = cp.asarray(qb_cpu, dtype=DTYPEf)
    qc_gpu = cp.zeros_like(qa_gpu)

    dflist = []
    for s in size:
        logging.info("########### processing size {} #############".format(s))

        #####################
        # numpy cpu
        thetime = benchmark(q4np.q4_mult, (qa_cpu[:s], qb_cpu[:s]), n_repeat=10, n_warmup=0)
        logging.info("q4np: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.cpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'CPU'
        tmp['implementation'] = 'numpy_CPU'
        tmp['function'] = 'q4_mult'
        dflist.append(tmp)

        #####################
        # numba cpu
        thetime = benchmark(q4nCPU.q4_mult, (qa_cpu[:s], qb_cpu[:s]), n_repeat=10, n_warmup=0)
        logging.info("q4nCPU: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.cpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'CPU'
        tmp['implementation'] = 'numba_CPU'
        tmp['function'] = 'q4_mult'
        dflist.append(tmp)

        #####################
        # cupy gpu
        thetime = benchmark(q4cp.q4_mult, (qa_gpu[:s], qb_gpu[:s], qc_gpu[:s]), n_repeat=10, n_warmup=0)
        logging.info("q4cp: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.gpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'GPU'
        tmp['implementation'] = 'cupy_GPU'
        tmp['function'] = 'q4_mult'
        dflist.append(tmp)

        #####################
        # numba gpu
        thetime = benchmark(q4nGPU.q4_mult, (qa_gpu[:s], qb_gpu[:s], qc_gpu[:s]), n_repeat=10, n_warmup=0)
        logging.info("q4nGPU: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.gpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'GPU'
        tmp['implementation'] = 'numba_GPU'
        tmp['function'] = 'q4_mult'
        dflist.append(tmp)

    df_qmult = pd.concat(dflist, ignore_index=True)
    df_qmult['gpu_mem'] = memGPUtot
    df_qmult['gpu_info'] = graphicscard
    df_qmult['cpu_info'] = cpu_info['brand_raw']
    df_qmult['system'] = system
    df_qmult['host'] = thehost
    df_qmult.to_excel('time_qmult.xlsx', index=False)

    return df_qmult

def plot_qmult_time(f='time_qmult.xlsx'):
    """
    Plot the graph of execution time for q4_mult function.
    """
    dftime = pd.read_excel(f)
    dftime.host = dftime.host.astype('category')
    dftime.system = dftime.system.astype('category')
    dftime.cpu_info = dftime.cpu_info.astype('category')
    dftime.gpu_info = dftime.gpu_info.astype('category')
    dftime.gpu_mem = dftime.gpu_mem.astype('category')
    dftime.function = dftime.function.astype('category')
    dftime.implementation = dftime.implementation.astype('category')
    dftime.type = dftime.type.astype('category')
    #dftime.ncrys = dftime.ncrys.astype('category')

    dftime.implementation = dftime.implementation.cat.reorder_categories(['numpy_CPU', 'numba_CPU', 'cupy_GPU', 'numba_GPU'])
    print(dftime.head())

    fs = 14
    boxprops = dict(linestyle='-', linewidth=3)
    medianprops = dict(linestyle='-', linewidth=3)
    capprops = dict(linestyle='-', linewidth=3)
    whiskerprops = dict(linestyle='-', linewidth=3)

    fig, axes = plt.subplots(1, 4, figsize=(10, 6), sharey=True, sharex=True)

    df = dftime[dftime.ncrys>=2**16*4]
    df.groupby('implementation').boxplot(column='timexec', by='ncrys', fontsize=fs, rot=90,
                                    boxprops=boxprops, medianprops=medianprops, capprops=capprops, whiskerprops=whiskerprops,
                                    ax=axes)
    axes[0].set_yscale('log')
    axes[0].set_ylim ([0.00002, 5])
    axes[0].set_ylabel(r'execution time, s', fontsize=fs)
    fig.suptitle('Quaternion multiplication', fontsize=20)
    plt.tight_layout()
    plt.savefig('time_qmult.pdf')
    plt.savefig('time_qmult.png', dpi=150)

    plt.show()

def time_qdisori(nmax=64*1024**2):
    """
    Timing elementary quaternion multiplication.
    """
    start = 1024
    stop = 64*1024*1024
    size = np.array([1024*2**i for i in range(20) if 1024*2**i <= stop], dtype=np.int32)[::-1]

    qa_cpu = q4np.q4_random(size[0])
    qb_cpu = qa_cpu[::-1,:]
    qsym_cpu = q4np.q4_sym_cubic()

    qa_gpu = cp.asarray(qa_cpu, dtype=DTYPEf)
    qb_gpu = cp.asarray(qb_cpu, dtype=DTYPEf)
    qc_gpu = cp.zeros_like(qa_gpu)
    qsym_gpu = cp.asarray(qsym_cpu, dtype=DTYPEf)
    a0_gpu = cp.zeros(qa_gpu.shape[0], dtype=DTYPEf)
    a1_gpu = cp.zeros_like(a0_gpu)

    dflist = []
    for s in size:
        logging.info("########### processing size {} #############".format(s))

        #####################
        # numpy cpu
        thetime = benchmark(q4np.q4_disori_angle, (qa_cpu[:s], qb_cpu[:s], qsym_cpu, 1), n_repeat=10, n_warmup=0)
        logging.info("q4np: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.cpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'CPU'
        tmp['implementation'] = 'numpy_CPU'
        tmp['function'] = 'q4_disori_angle'
        dflist.append(tmp)

        #####################
        # numba cpu
        thetime = benchmark(q4nCPU.q4_disori_angle, (qa_cpu[:s], qb_cpu[:s], qsym_cpu, 1), n_repeat=10, n_warmup=0)
        logging.info("q4nCPU: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.cpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'CPU'
        tmp['implementation'] = 'numba_CPU'
        tmp['function'] = 'q4_disori_angle'
        dflist.append(tmp)

        #####################
        # cupy gpu
        thetime = benchmark(q4cp.q4_disori_angle, (qa_gpu[:s], qb_gpu[:s], qc_gpu[:s], qsym_gpu, a0_gpu[:s], a1_gpu[:s], 1, True), n_repeat=10, n_warmup=0)
        logging.info("q4cp: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.gpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'GPU'
        tmp['implementation'] = 'cupy_GPU'
        tmp['function'] = 'q4_disori_angle'
        dflist.append(tmp)

        #####################
        # numba gpu
        thetime = benchmark(q4nGPU.q4_disori_angle, (qa_gpu[:s], qb_gpu[:s], qsym_gpu, a0_gpu[:s]), n_repeat=10, n_warmup=0)
        logging.info("q4nGPU: {}".format(thetime))

        tmp = pd.DataFrame(columns=['host', 'system', 'cpu_info', 'gpu_info', 'gpu_mem',
                                    'function', 'implementation', 'type', 'ncrys', 'timexec'])
        tmp['timexec'] = thetime.gpu_times.flatten()
        tmp['ncrys'] = s
        tmp['type'] = 'GPU'
        tmp['implementation'] = 'numba_GPU'
        tmp['function'] = 'q4_disori_angle'
        dflist.append(tmp)

    df_qdisori = pd.concat(dflist, ignore_index=True)
    df_qdisori['gpu_mem'] = memGPUtot
    df_qdisori['gpu_info'] = graphicscard
    df_qdisori['cpu_info'] = cpu_info['brand_raw']
    df_qdisori['system'] = system
    df_qdisori['host'] = thehost
    df_qdisori.to_excel('time_qdisori.xlsx', index=False)

    return df_qdisori

def plot_qdisori_time(f='time_qdisori.xlsx'):
    """
    Plot the graph of execution time for q4_mult function.
    """
    dftime = pd.read_excel(f)
    dftime.host = dftime.host.astype('category')
    dftime.system = dftime.system.astype('category')
    dftime.cpu_info = dftime.cpu_info.astype('category')
    dftime.gpu_info = dftime.gpu_info.astype('category')
    dftime.gpu_mem = dftime.gpu_mem.astype('category')
    dftime.function = dftime.function.astype('category')
    dftime.implementation = dftime.implementation.astype('category')
    dftime.type = dftime.type.astype('category')
    #dftime.ncrys = dftime.ncrys.astype('category')

    dftime.implementation = dftime.implementation.cat.reorder_categories(['numpy_CPU', 'numba_CPU', 'cupy_GPU', 'numba_GPU'])
    print(dftime.head())

    fs = 14
    boxprops = dict(linestyle='-', linewidth=3)
    medianprops = dict(linestyle='-', linewidth=3)
    capprops = dict(linestyle='-', linewidth=3)
    whiskerprops = dict(linestyle='-', linewidth=3)

    fig, axes = plt.subplots(1, 4, figsize=(10, 6), sharey=True, sharex=True)

    df = dftime[dftime.ncrys>=2**16*4]
    df.groupby('implementation').boxplot(column='timexec', by='ncrys', fontsize=fs, rot=90,
                                    boxprops=boxprops, medianprops=medianprops, capprops=capprops, whiskerprops=whiskerprops,
                                    ax=axes)
    axes[0].set_yscale('log')
    axes[0].set_ylim ([0.0005, 50])
    axes[0].set_ylabel(r'execution time, s', fontsize=fs)
    fig.suptitle('Quaternion disorientation', fontsize=20)
    plt.tight_layout()
    plt.savefig('time_qdisori.pdf')
    plt.savefig('time_qdisori.png', dpi=150)

    plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

