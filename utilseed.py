import random
import numpy as np
import torch
import os

def set_seed(seed=2025):
    """
    设置全局随机种子，确保实验可复现
    参数:
        seed: 随机种子值，默认使用当前年份2025
    """
    # 设置Python和系统环境
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置NumPy
    np.random.seed(seed)
    
    # 设置PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置cuDNN保证确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置TensorFlow（如果使用）
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # print(f"全局随机种子已设置为: {seed}")