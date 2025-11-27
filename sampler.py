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

# -----------------------