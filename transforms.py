import numpy as np
from scipy.fftpack import dct, idct

# --- 成員 C 的工作區域 ---

def forward_dct_block(block):
    """
    對 8x8 區塊進行 DCT 轉換
    目前使用 scipy 實作
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def inverse_dct_block(block):
    """
    對 8x8 區塊進行 IDCT 轉換
    目前使用 scipy 實作
    """
    # 目前先用 scipy 的 IDCT 當作標準答案
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# -----------------------