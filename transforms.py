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


# 方法 1: 直接數學實作 (Direct Matrix Multiplication)
# 預先計算 cosine 係數矩陣，使用矩陣乘法加速

_IDCT_BASIS = None

def _get_idct_basis():
    """預先計算 IDCT 的 cosine 基底矩陣 (8x8)"""
    global _IDCT_BASIS
    if _IDCT_BASIS is None:
        basis = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if i == 0:
                    basis[i, j] = 1.0 / np.sqrt(8)
                else:
                    basis[i, j] = np.sqrt(2.0 / 8) * np.cos((2*j + 1) * i * np.pi / 16)
        _IDCT_BASIS = basis
    return _IDCT_BASIS


def inverse_dct_block_direct(block):
    """
    使用預先計算的基底矩陣進行 IDCT
    IDCT(X) = B^T * X * B
    其中 B 是 DCT 基底矩陣
    """
    basis = _get_idct_basis()
    return basis.T @ block @ basis


# 方法 2: 分離式 1D IDCT (Separable 1D IDCT)
# 先對行做 IDCT，再對列做 IDCT

def _idct_1d(x):
    """
    對單一維度（8 個元素的向量）進行 IDCT
    使用直接公式計算
    """
    N = 8
    result = np.zeros(N)
    for n in range(N):
        sum_val = 0.0
        for k in range(N):
            if k == 0:
                alpha_k = 1.0 / np.sqrt(2)
            else:
                alpha_k = 1.0
            sum_val += alpha_k * x[k] * np.cos((2*n + 1) * k * np.pi / (2*N))
        result[n] = sum_val * np.sqrt(2.0 / N)
    return result


def inverse_dct_block_separable(block):
    """
    使用分離式 1D IDCT
    先對每一列做 IDCT，再對每一行做 IDCT
    """
    # 對每一列做 IDCT
    temp = np.zeros((8, 8))
    for i in range(8):
        temp[i, :] = _idct_1d(block[i, :])
    
    # 對每一行做 IDCT
    result = np.zeros((8, 8))
    for j in range(8):
        result[:, j] = _idct_1d(temp[:, j])
    
    return result


# 方法 3: 使用查表法優化 (Look-Up Table)
# 預先計算所有可能的 cosine 值

_COSINE_LUT = None

def _get_cosine_lut():
    """預先計算所有需要的 cosine 值"""
    global _COSINE_LUT
    if _COSINE_LUT is None:
        lut = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                lut[i, j] = np.cos((2*j + 1) * i * np.pi / 16)
        _COSINE_LUT = lut
    return _COSINE_LUT


def inverse_dct_block_lut(block):
    """
    使用查表法的 IDCT 實作
    預先計算 cosine 值以加速運算
    """
    cos_lut = _get_cosine_lut()
    result = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for u in range(8):
                for v in range(8):
                    # Alpha 係數
                    alpha_u = 1.0 / np.sqrt(2) if u == 0 else 1.0
                    alpha_v = 1.0 / np.sqrt(2) if v == 0 else 1.0
                    
                    # 使用查表的 cosine 值
                    sum_val += (alpha_u * alpha_v * block[u, v] * 
                               cos_lut[u, i] * cos_lut[v, j])
            
            result[i, j] = sum_val * 0.25  # 正規化係數 = 2/8 * 2/8 = 1/4
    
    return result


# 方法 4: 向量化版本 (Vectorized)
# 使用 NumPy 的向量化操作，一次計算整個矩陣

_IDCT_MATRIX = None

def _get_idct_matrix():
    """產生 64x64 的 IDCT 轉換矩陣"""
    global _IDCT_MATRIX
    if _IDCT_MATRIX is None:
        matrix = np.zeros((64, 64))
        for i in range(8):
            for j in range(8):
                for u in range(8):
                    for v in range(8):
                        alpha_u = 1.0 / np.sqrt(2) if u == 0 else 1.0
                        alpha_v = 1.0 / np.sqrt(2) if v == 0 else 1.0
                        
                        cos_u = np.cos((2*i + 1) * u * np.pi / 16)
                        cos_v = np.cos((2*j + 1) * v * np.pi / 16)
                        
                        row_idx = i * 8 + j
                        col_idx = u * 8 + v
                        matrix[row_idx, col_idx] = 0.25 * alpha_u * alpha_v * cos_u * cos_v
        
        _IDCT_MATRIX = matrix
    return _IDCT_MATRIX


def inverse_dct_block_vectorized(block):
    """
    完全向量化的 IDCT
    將 8x8 block 展平為 64 維向量，用矩陣乘法一次完成
    """
    idct_matrix = _get_idct_matrix()
    block_flat = block.flatten()
    result_flat = idct_matrix @ block_flat
    return result_flat.reshape((8, 8))


# 方法 5: 使用 FFT 加速 (FFT-based)
# 利用 DCT 和 FFT 的關係來加速計算

def inverse_dct_block_fft(block):
    """
    使用 FFT 來實作 IDCT
    DCT 可以透過 FFT 來高效計算
    """
    # 對行方向先做 IDCT (使用 FFT)
    temp = np.zeros((8, 8))
    for i in range(8):
        temp[i, :] = _idct_1d_fft(block[i, :])
    
    # 對列方向做 IDCT
    result = np.zeros((8, 8))
    for j in range(8):
        result[:, j] = _idct_1d_fft(temp[:, j])
    
    return result


def _idct_1d_fft(x):
    """
    使用 FFT 實作 1D IDCT
    IDCT 可以表示為特殊的 FFT
    """
    N = len(x)
    
    # 建構擴展序列
    y = np.zeros(2 * N)
    for k in range(N):
        if k == 0:
            y[0] = x[0] / np.sqrt(2)
        else:
            y[k] = x[k]
            y[2*N - k] = x[k]
    
    # 執行 FFT
    Y = np.fft.fft(y)
    
    # 取實部並調整
    result = np.real(Y[:N]) * np.sqrt(2.0 / N)
    
    return result


# 方法 6: AAN 快速演算法簡化版
# Arai, Agui, and Nakajima 演算法（簡化版）

# AAN 演算法的縮放因子
_AAN_SCALE = None

def _get_aan_scale():
    """取得 AAN 演算法的縮放因子"""
    global _AAN_SCALE
    if _AAN_SCALE is None:
        scale = np.ones(8)
        scale[0] = 1.0 / (2.0 * np.sqrt(2))
        for i in range(1, 8):
            scale[i] = 1.0 / (4.0 * np.cos(i * np.pi / 16))
        _AAN_SCALE = scale
    return _AAN_SCALE


def inverse_dct_block_aan(block):
    """
    使用 AAN 縮放因子的快速 IDCT（向量化版）
      - 先用 row/col 的縮放因子，把頻域係數做預先 scale
      - 再用預先計好的 DCT 基底做 B^T * X * B
      - 不用 Python 的雙層 for，全部交給 NumPy 做矩陣運算
    """
    scale = _get_aan_scale()              # shape: (8,)
    basis = _get_idct_basis()             # shape: (8, 8)

    # 利用 outer product 一次產生 8x8 的縮放矩陣
    scale_mat = scale[:, None] * scale[None, :]   # (8,1) * (1,8) -> (8,8)

    # 元素乘，取代原來的雙層 for 迴圈
    scaled_block = block * scale_mat

    # 矩陣形式的 IDCT：B^T * scaled * B
    return basis.T @ scaled_block @ basis

_IDCT_1D_MATRIX = None

def _get_idct_1d_matrix():
    """
    產生 8x8 的 1D IDCT 轉換矩陣 C
    
    對應 scipy 的 idct(x, norm='ortho')：
      y[n] = sum_k alpha_k * x[k] * cos( (2n+1)kπ / (2N) )
      其中 alpha_0 = sqrt(1/N), alpha_k = sqrt(2/N) (k>0)
    """
    global _IDCT_1D_MATRIX
    if _IDCT_1D_MATRIX is None:
        N = 8
        C = np.zeros((N, N), dtype=np.float64)
        for n in range(N):
            for k in range(N):
                if k == 0:
                    alpha = np.sqrt(1.0 / N)
                else:
                    alpha = np.sqrt(2.0 / N)
                C[n, k] = alpha * np.cos((np.pi * (2*n + 1) * k) / (2 * N))
        _IDCT_1D_MATRIX = C
    return _IDCT_1D_MATRIX


# 方法 7: 完全矩陣化版本 (Matrix Multiplication)
# 使用完全矩陣化的 2D IDCT 實作

def inverse_dct_block_matmul(block):
    """
    Y = C * X * C^T
    其中 C 為 8x8 1D IDCT 轉換矩陣。
    """
    C = _get_idct_1d_matrix()
    return C @ block @ C.T



# 效能測試函數
def benchmark_idct_methods(test_block=None, iterations=1000):
    """
    測試各種 IDCT 方法的效能與精度
    
    - 以 scipy 版 inverse_dct_block 當作 ground truth
    - 每個方法重複跑 iterations 次，測量平均時間
    - 計算：
        * 最大絕對誤差 (max |diff|)
        * 平均絕對誤差 (mean |diff|)
    """
    import time

    if test_block is None:
        np.random.seed(42)
        test_block = np.random.randn(8, 8) * 100  # 模擬 DCT 係數

    # 方法表：名稱 → 函式
    methods = {
        'baseline (scipy)': inverse_dct_block,
        'direct':           inverse_dct_block_direct,
        'separable':        inverse_dct_block_separable,
        'lut':              inverse_dct_block_lut,
        'vectorized64':     inverse_dct_block_vectorized,
        'fft':              inverse_dct_block_fft,
        'aan':              inverse_dct_block_aan,
        'matmul':           inverse_dct_block_matmul,
    }

    results = {}
    baseline_result = None

    print("=" * 70)
    print("IDCT 方法效能 / 精度測試（單一 8x8 block）")
    print("=" * 70)

    for name, func in methods.items():
        _ = func(test_block)

        start = time.perf_counter()
        for _ in range(iterations):
            result = func(test_block)
        elapsed = (time.perf_counter() - start) * 1000.0  # ms

        if baseline_result is None:
            # 第一個 baseline 當作 ground truth
            baseline_result = result
            max_err = 0.0
            mean_err = 0.0
        else:
            diff = result - baseline_result
            max_err = float(np.max(np.abs(diff)))
            mean_err = float(np.mean(np.abs(diff)))

        results[name] = {
            'time_ms_total': elapsed,
            'time_ms_per_call': elapsed / iterations,
            'max_err': max_err,
            'mean_err': mean_err,
        }

        print(f"{name:15s} | "
              f"總時間: {elapsed:8.3f} ms | "
              f"單次: {elapsed/iterations:8.5f} ms | "
              f"max_err: {max_err:9.2e} | "
              f"mean_err: {mean_err:9.2e}")

    print("=" * 70)

    # 依照時間排序
    fastest = sorted(results.items(), key=lambda x: x[1]['time_ms_per_call'])
    print("\n按單次時間排序（由快到慢）：")
    for name, r in fastest:
        print(f"  {name:15s}  {r['time_ms_per_call']:8.5f} ms")

    # 依照誤差排序（排除 baseline）
    non_baseline = [(n, r) for n, r in results.items() if 'baseline' not in n]
    most_accurate = sorted(non_baseline, key=lambda x: (x[1]['max_err'], x[1]['mean_err']))
    print("\n按精度排序（max_err → mean_err，由準到不準）：")
    for name, r in most_accurate:
        print(f"  {name:15s}  max_err={r['max_err']:9.2e}, mean_err={r['mean_err']:9.2e}")

    print("=" * 70)
    return results