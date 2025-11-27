import numpy as np
import time

def rgb_to_ycbcr(img):
    """將 RGB 圖片轉換為 YCbCr"""
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = img.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128 # Shift chroma
    return ycbcr

def ycbcr_to_rgb(img):
    """將 YCbCr 圖片轉換回 RGB"""
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = img.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)

def make_blocks(channel):
    """將 2D 陣列切割成 8x8 的區塊 (Blocks)"""
    h, w = channel.shape
    # 簡單起見，這裡假設圖片長寬已補滿為 8 的倍數
    # 實際專案可能需要 Padding 函式
    blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blocks.append(channel[i:i+8, j:j+8])
    return np.array(blocks), h, w

def reconstruct_from_blocks(blocks, h, w):
    """將 8x8 區塊組回 2D 陣列"""
    img = np.zeros((h, w))
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            img[i:i+8, j:j+8] = blocks[idx]
            idx += 1
    return img

def calculate_psnr(img1, img2):
    """
    計算峰值訊號雜訊比 (PSNR)
    數值越高代表兩張圖越像 (通常 > 30dB 代表品質不錯)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def count_non_zero(blocks_list):
    """計算一堆 block 中，不為 0 的數值有多少個"""
    total_non_zero = 0
    for block in blocks_list:
        total_non_zero += np.count_nonzero(block)
    return total_non_zero

def estimate_compression_ratio(original_shape, nz_count_y, nz_count_c):
    """
    估算壓縮率
    original_shape: (H, W, 3)
    nz_count: 非零係數總數
    """
    h, w, _ = original_shape
    original_bytes = h * w * 3  # 原始 RGB 大小 (bytes)
    
    # --- 估算邏輯 ---
    # 假設每個非零係數平均需要 1.5 bytes (包含數值本身 + RLE 編碼開銷)
    # 這是一個經驗法則估計值，並非精確的 Huffman 編碼長度
    estimated_compressed_bytes = (nz_count_y + nz_count_c) * 1.5 
    
    # 加上一些檔頭 (Header) 的基本開銷，假設 600 bytes
    estimated_compressed_bytes += 600
    
    ratio = original_bytes / estimated_compressed_bytes
    return original_bytes, estimated_compressed_bytes, ratio