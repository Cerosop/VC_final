import numpy as np

# ==========================================
# Part 1: Downsampling (降樣)
# ==========================================

def downsample(channel, ratio=0.5):
    """
    [Baseline] Bilinear Interpolation (雙線性插值)
    您的指定基準：使用雙線性插值進行縮圖。
    """
    h, w = channel.shape
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # 1. 產生目標圖的網格座標 (對應回原圖的浮點座標)
    # 中心對齊公式: src = (dst + 0.5) / scale - 0.5
    x_grid = (np.arange(new_w) + 0.5) / ratio - 0.5
    y_grid = (np.arange(new_h) + 0.5) / ratio - 0.5
    
    # 2. 限制邊界 (避免負數)
    x_grid = np.maximum(x_grid, 0)
    y_grid = np.maximum(y_grid, 0)
    
    # 3. 取整數座標 (x0, y0) 與 下一個座標 (x1, y1)
    x0 = np.floor(x_grid).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    
    y0 = np.floor(y_grid).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    
    # 4. 計算權重 (小數部分)
    # reshape 用於廣播: wx 變 (1, W), wy 變 (H, 1)
    wx = (x_grid - x0).reshape(1, -1)
    wy = (y_grid - y0).reshape(-1, 1)
    
    # 5. 取出四個點的值 (Vectorized Lookup)
    # 利用 numpy 的廣播機制一次取出所有像素
    # Ia:左上, Ib:左下, Ic:右上, Id:右下
    Ia = channel[y0[:, None], x0]
    Ib = channel[y1[:, None], x0]
    Ic = channel[y0[:, None], x1]
    Id = channel[y1[:, None], x1]
    
    # 6. 雙線性插值公式
    result = (Ia * (1 - wx) * (1 - wy) +
              Ic * wx       * (1 - wy) +
              Ib * (1 - wx) * wy +
              Id * wx       * wy)
              
    return np.clip(result, 0, 255).astype(channel.dtype)

def downsample_nearest(channel, ratio=0.5):
    """
    [Method A] Nearest Neighbor (最近鄰)
    比較用：速度最快，但會有鋸齒。
    """
    step = int(1 / ratio)
    return channel[::step, ::step]

def downsample_average(channel, ratio=0.5):
    """
    [Method B] Area Average (區域平均)
    比較用：YUV420 標準作法，畫質通常優於 Bilinear (較少混疊)。
    """
    h, w = channel.shape
    if ratio == 0.5:
        # 確保偶數長寬
        h_even, w_even = h - (h % 2), w - (w % 2)
        cut = channel[:h_even, :w_even]
        # 2x2 區塊取平均
        return cut.reshape(h_even//2, 2, w_even//2, 2).mean(axis=(1, 3)).astype(channel.dtype)
    else:
        # 非 0.5 比例時退回 nearest
        return downsample_nearest(channel, ratio)


# ==========================================
# Part 2: Upsampling (升樣)
# ==========================================

def upsample(channel, target_h, target_w):
    """
    [Baseline] Nearest Neighbor (最近鄰)
    您的指定基準：直接複製像素，速度最快，但有馬賽克。
    """
    h, w = channel.shape
    
    # 計算來源座標 (反向映射)
    # 例如目標是 0, 1, 2... 對應回原圖是 0, 0, 1...
    row_idx = (np.arange(target_h) * (h / target_h)).astype(int)
    col_idx = (np.arange(target_w) * (w / target_w)).astype(int)
    
    # 防止越界
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    
    # 使用 Numpy 的 Fancy Indexing 快速取出
    return channel[row_idx[:, None], col_idx]

def upsample_bilinear(channel, target_h, target_w):
    """
    [Method A] Bilinear Interpolation (雙線性插值)
    比較用：比 Baseline 平滑，消除馬賽克，但邊緣變糊。
    """
    h, w = channel.shape
    
    x_grid = (np.arange(target_w) + 0.5) * (w / target_w) - 0.5
    y_grid = (np.arange(target_h) + 0.5) * (h / target_h) - 0.5
    
    x_grid = np.clip(x_grid, 0, w - 1)
    y_grid = np.clip(y_grid, 0, h - 1)
    
    x0 = np.floor(x_grid).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y_grid).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wx = (x_grid - x0).reshape(1, -1)
    wy = (y_grid - y0).reshape(-1, 1)
    
    Ia = channel[y0[:, None], x0]
    Ib = channel[y1[:, None], x0]
    Ic = channel[y0[:, None], x1]
    Id = channel[y1[:, None], x1]
    
    result = (Ia * (1-wx) * (1-wy) +
              Ic * wx     * (1-wy) +
              Ib * (1-wx) * wy     +
              Id * wx     * wy)
              
    return np.clip(result, 0, 255).astype(channel.dtype)

def upsample_bicubic(channel, target_h, target_w):
    """
    [Method B] Bicubic Interpolation (雙立方插值 - 針對 2x 優化)
    比較用：比 Bilinear 銳利，運算最慢。
    """
    h, w = channel.shape
    
    # 僅針對 2 倍放大做優化實作，若非 2 倍則退回 bilinear
    if target_h != h * 2 or target_w != w * 2:
        return upsample_bilinear(channel, target_h, target_w)

    # 這裡使用 Catmull-Rom Spline 權重模擬卷積
    padded = np.pad(channel, ((1, 2), (1, 2)), mode='symmetric').astype(np.float32)
    out = np.zeros((target_h, target_w), dtype=np.float32)
    
    # 1. 水平方向插值
    temp_w = np.zeros((h, target_w), dtype=np.float32)
    temp_w[:, 0::2] = channel # 偶數點
    
    p0 = padded[1:-2, 0:-3]; p1 = padded[1:-2, 1:-2]
    p2 = padded[1:-2, 2:-1]; p3 = padded[1:-2, 3:]
    temp_w[:, 1::2] = (-p0 + 9*p1 + 9*p2 - p3) / 16.0 # 奇數點
    
    # 2. 垂直方向插值
    temp_pad = np.pad(temp_w, ((1, 2), (0, 0)), mode='symmetric')
    out[0::2, :] = temp_w # 偶數行
    
    tp0 = temp_pad[0:-3, :]; tp1 = temp_pad[1:-2, :]
    tp2 = temp_pad[2:-1, :]; tp3 = temp_pad[3:, :]
    out[1::2, :] = (-tp0 + 9*tp1 + 9*tp2 - tp3) / 16.0 # 奇數行
    
    return np.clip(out, 0, 255).astype(channel.dtype)