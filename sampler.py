import numpy as np
import cv2 # 為了方便示範，暫用 opencv resize，成員 B 需手寫

# --- 成員 B 的工作區域 ---

def downsample(channel, ratio=0.5):
    """
    色度抽樣 (4:2:0 通常是長寬各減半)
    """
    h, w = channel.shape
    new_h, new_w = int(h * ratio), int(w * ratio)
    # 這裡可以用簡單的平均或丟棄法
    return cv2.resize(channel, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def upsample(channel, target_h, target_w):
    """
    色度升樣（標準作法：最近鄰插值 Nearest Neighbor）
    例如 4:2:0 中，Cb/Cr 一個像素複製到 2x2 的 Y 區域。
    """
    h, w = channel.shape
    if h == target_h and w == target_w:
        return channel

    # 通用最近鄰：對目標座標反投影到原始座標並取 floor
    row_idx = (np.floor(np.arange(target_h) * (h / target_h))).astype(int)
    col_idx = (np.floor(np.arange(target_w) * (w / target_w))).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return channel[row_idx][:, col_idx]

def upsample_bilinear(channel, target_h, target_w):
    """
    色度升樣 B Bilinear Interpolation (高效能 Numpy 版)
    """
    h, w = channel.shape
    if h == target_h and w == target_w:
        return channel
    # 1. 產生網格座標 (Vectorized)
    # 對應你原本的 for x_t, for y_t
    x = np.linspace(0, w - 1, target_w)
    y = np.linspace(0, h - 1, target_h)
    
    # 2. 找到最近的整數座標 (x1, y1, x2, y2)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    
    # 3. 計算權重 (Broadcasting)
    # 對應你原本的 x_diff, y_diff
    wx = (x - x0).reshape(1, -1) # 形狀變 (1, W)
    wy = (y - y0).reshape(-1, 1) # 形狀變 (H, 1)
    
    # 4. 一次取出所有像素值 (Fancy Indexing)
    # 對應你原本的 p11, p12...
    Ia = channel[y0[:, None], x0] # Top-Left
    Ib = channel[y1[:, None], x0] # Bottom-Left
    Ic = channel[y0[:, None], x1] # Top-Right
    Id = channel[y1[:, None], x1] # Bottom-Right
    
    # 5. 一次算出整張圖的結果
    result = (Ia * (1 - wx) * (1 - wy) +
              Ic * wx       * (1 - wy) +
              Ib * (1 - wx) * wy +
              Id * wx       * wy)
              
    return np.clip(result, 0, 255).astype(channel.dtype)