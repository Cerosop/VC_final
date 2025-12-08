import numpy as np

# --- 成員 A 的工作區域 ---

def quantize(block, q_table):
    """量化：DCT 係數除以量化表"""
    return np.round(block / q_table)

def dequantize(block, q_table):
    """反量化：係數乘以量化表"""
    return block * q_table


# 基本 Zigzag 掃描順序 (8x8)
_ZZ_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]

def _zigzag_flatten(block8x8: np.ndarray):
    coeffs = []
    for r,c in _ZZ_IDX:
        coeffs.append(int(block8x8[r, c]))
    return coeffs

def _inverse_zigzag(coeffs):
    out = np.zeros((8, 8), dtype=np.int32)
    for (r, c), v in zip(_ZZ_IDX, coeffs):
        out[r, c] = v
    return out


# 全圖共用 Huffman 的全域狀態（由 prepare_huff_global 建立）
_GLOBAL_HUFF_ROOT = None
_GLOBAL_HUFF_CODES = None


# JPEG-like DC 差分用的全域狀態
_DC_PREDICTORS = {"Y": 0, "Cb": 0, "Cr": 0}
_CURRENT_COMPONENT = "Y"  # 由 main.py 在處理 Y/Cb/Cr 前設定


# 簡單 Huffman 樹
class _Node:
    __slots__ = ('left','right','sym','freq')
    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym = sym
        self.freq = freq
        self.left = left
        self.right = right

def _build_huffman(freqs):
    nodes = [_Node(sym=s, freq=f) for s, f in freqs.items()]
    if not nodes:
        # 退化情況：至少給一個 EOB
        nodes = [_Node(sym=('EOB',), freq=1)]
    while len(nodes) > 1:
        nodes.sort(key=lambda n: n.freq)
        a = nodes.pop(0)
        b = nodes.pop(0)
        parent = _Node(sym=None, freq=a.freq + b.freq, left=a, right=b)
        nodes.append(parent)
    return nodes[0]

def _gen_codes(node, prefix='', out=None):
    if out is None:
        out = {}
    if node.sym is not None:
        out[node.sym] = prefix or '0'  # 單一節點時給 '0'
        return out
    _gen_codes(node.left, prefix + '0', out)
    _gen_codes(node.right, prefix + '1', out)
    return out


def _rle_encode(coeffs):
    """對 Zigzag 係數做零值游程編碼，回傳符號清單。
    符號為 (run, value) 或 ('EOB',)。
    """
    symbols = []
    run = 0
    # 逐項直到最後一個非零，尾端補 EOB
    last_nz = 0
    for i, v in enumerate(coeffs):
        if v != 0:
            last_nz = i
    for i, v in enumerate(coeffs[:last_nz+1]):
        if v == 0:
            run += 1
        else:
            symbols.append((run, int(v)))
            run = 0
    # 尾端零以 EOB 表示
    if last_nz < 63:
        symbols.append(('EOB',))
    return symbols

def _rle_decode(symbols):
    coeffs = []
    for s in symbols:
        if s == ('EOB',):
            # 尾端補零到 64
            while len(coeffs) < 64:
                coeffs.append(0)
            break
        run, val = s
        coeffs.extend([0] * run)
        coeffs.append(int(val))
        if len(coeffs) >= 64:
            coeffs = coeffs[:64]
            break
    if len(coeffs) < 64:
        coeffs.extend([0] * (64 - len(coeffs)))
    return coeffs


def reset_dc_predictors():
    """
    將 DC predictor 重設為 0。
    需在每張圖開始（編碼與解碼各一次）呼叫。
    """
    global _DC_PREDICTORS
    _DC_PREDICTORS = {"Y": 0, "Cb": 0, "Cr": 0}


def set_current_component(name: str):
    """
    設定目前處理的分量（'Y' / 'Cb' / 'Cr'）。
    由 main.py 在處理各 channel 迴圈前呼叫。
    """
    global _CURRENT_COMPONENT
    if name in ("Y", "Cb", "Cr"):
        _CURRENT_COMPONENT = name


def prepare_huff_global(blocks):
    """
    根據整張圖（所有量化後 8x8 block）的 RLE 符號建立一棵「全圖共用」的 Huffman 樹。
    會將結果儲存在模組全域變數 _GLOBAL_HUFF_ROOT / _GLOBAL_HUFF_CODES 中。
    """
    global _GLOBAL_HUFF_ROOT, _GLOBAL_HUFF_CODES

    freqs = {}
    for block in blocks:
        block = np.asarray(block)
        zz = _zigzag_flatten(block)
        symbols = _rle_encode(zz)
        for s in symbols:
            freqs[s] = freqs.get(s, 0) + 1

    root = _build_huffman(freqs)
    codes = _gen_codes(root)

    _GLOBAL_HUFF_ROOT = root
    _GLOBAL_HUFF_CODES = codes

# -------------------------------------------------------------------------
# Method 1: Huffman (原版，Per-block Huffman)
# -------------------------------------------------------------------------

def encode_block_huff(block):
    """
    將量化後的 8x8 區塊轉為 Bitstream：
    - Zigzag -> RLE(零) -> 建立 Huffman Tree -> 逐 bit 編碼
    回傳：{"codec":"huff","tree":root_node,"bits":str}
    """
    block = np.asarray(block)
    zz = _zigzag_flatten(block)
    symbols = _rle_encode(zz)

    # 建頻率表
    freqs = {}
    for s in symbols:
        freqs[s] = freqs.get(s, 0) + 1
    root = _build_huffman(freqs)
    codes = _gen_codes(root)

    bits = ''.join(codes[s] for s in symbols)
    return {
        'codec': 'huff',
        'tree': root,
        'bits': bits,
    }

def decode_block_huff(stream_data):
    """
    將 Bitstream 解碼回 8x8 量化區塊：
    - 逐 bit 讀取，沿 Huffman 樹遍歷至葉節點取得符號
    - 反 RLE -> 反 Zigzag
    """
    # 兼容舊格式：若不是 dict，直接視為已是量化區塊
    if not isinstance(stream_data, dict):
        return stream_data

    codec = stream_data.get('codec')
    # 若不是 huff / huff_global 編碼，直接回傳（讓其他 decode_* 處理）
    if codec not in ('huff', 'huff_global'):
        return stream_data

    root = stream_data['tree']
    bits = stream_data['bits']

    # 特殊情況：退化 Huffman 樹（只有一個符號）
    # 這時 root 是葉節點，沒有 left/right，可以直接依 bits 長度複製同一個符號。
    if getattr(root, "left", None) is None and getattr(root, "right", None) is None:
        if not bits:
            symbols = [root.sym]
        else:
            symbols = [root.sym] * len(bits)
    else:
        # 一般情況：bit-by-bit 順著樹走
        symbols = []
        node = root
        for b in bits:
            node = node.left if b == '0' else node.right
            # 安全檢查：若 bit 序列異常導致走到 None，提前中止
            if node is None:
                break
            if node.sym is not None:
                symbols.append(node.sym)
                node = root
                # 提前終止：遇到 EOB 且已足夠
                if symbols[-1] == ('EOB',):
                    break

    coeffs = _rle_decode(symbols)
    block8 = _inverse_zigzag(coeffs)
    return block8


def encode_block_huff_global(block):
    """
    使用「全圖共用」的 Huffman 樹進行編碼。
    需先呼叫 prepare_huff_global 建立 _GLOBAL_HUFF_ROOT / _GLOBAL_HUFF_CODES；
    若尚未建立，則退回到一般 per-block Huffman。
    """
    global _GLOBAL_HUFF_ROOT, _GLOBAL_HUFF_CODES

    block = np.asarray(block)

    if _GLOBAL_HUFF_ROOT is None or _GLOBAL_HUFF_CODES is None:
        # 尚未準備好全域 Huffman，退回 per-block 版本避免崩潰
        return encode_block_huff(block)

    zz = _zigzag_flatten(block)
    symbols = _rle_encode(zz)

    bits = ''.join(_GLOBAL_HUFF_CODES[s] for s in symbols)
    return {
        'codec': 'huff_global',
        'tree': _GLOBAL_HUFF_ROOT,
        'bits': bits,
    }


def decode_block_huff_global(stream_data):
    """
    Global Huffman 版的解碼，實作上直接重用 decode_block_huff。
    """
    return decode_block_huff(stream_data)


# -------------------------------------------------------------------------
# Method 5: Huffman with DC DPCM (JPEG-like, DC/AC 分離於預處理階段)
# -------------------------------------------------------------------------


def _apply_dc_dpcm_encode(coeffs):
    """
    對 Zigzag 展開後的 64 維係數做 DC 差分：
    - coeffs[0] 會被替換成 (DC - predictor)，並更新 predictor。
    - AC 係數維持不變。
    """
    global _DC_PREDICTORS, _CURRENT_COMPONENT
    dc = coeffs[0]
    pred = _DC_PREDICTORS.get(_CURRENT_COMPONENT, 0)
    diff = int(dc - pred)
    _DC_PREDICTORS[_CURRENT_COMPONENT] = int(dc)
    out = list(coeffs)
    out[0] = diff
    return out


def _apply_dc_dpcm_decode(coeffs):
    """
    對 Zigzag 展開後的 64 維係數做 DC 差分還原：
    - coeffs[0] 視為 (DC - predictor)，恢復為真正的 DC 並更新 predictor。
    """
    global _DC_PREDICTORS, _CURRENT_COMPONENT
    diff = int(coeffs[0])
    pred = _DC_PREDICTORS.get(_CURRENT_COMPONENT, 0)
    dc = pred + diff
    _DC_PREDICTORS[_CURRENT_COMPONENT] = int(dc)
    out = list(coeffs)
    out[0] = dc
    return out


def encode_block_huff_dpcm(block):
    """
    Huffman + DC DPCM：
    - Zigzag 展開後，先將 DC 係數做差分 (DPCM)，只對 AC 維持原本統計特性。
    - 再將整個 64 維序列丟入 RLE + Huffman（每個 block 各自建樹）。

    注意：為了簡化實作與維持與其他 method 相容，
    我們仍以 per-block Huffman 編碼，僅在 DC 預處理階段做 DPCM。
    """
    block = np.asarray(block)
    zz = _zigzag_flatten(block)
    zz_dpcm = _apply_dc_dpcm_encode(zz)
    symbols = _rle_encode(zz_dpcm)

    # 建頻率表（per-block）
    freqs = {}
    for s in symbols:
        freqs[s] = freqs.get(s, 0) + 1
    root = _build_huffman(freqs)
    codes = _gen_codes(root)

    bits = ''.join(codes[s] for s in symbols)
    return {
        'codec': 'huff_dpcm',
        'tree': root,
        'bits': bits,
    }


def decode_block_huff_dpcm(stream_data):
    """
    Huffman + DC DPCM 的解碼：
    - 與 decode_block_huff 類似，先還原出 64 維序列（其第 0 項為 DC diff）
    - 再依照 DC predictor 還原真正 DC，最後反 Zigzag 成 8x8 block。
    """
    # 兼容舊格式：若不是 dict，直接視為已是量化區塊
    if not isinstance(stream_data, dict):
        return stream_data

    if stream_data.get('codec') != 'huff_dpcm':
        return stream_data

    root = stream_data['tree']
    bits = stream_data['bits']

    # 處理可能的退化 Huffman 樹
    if getattr(root, "left", None) is None and getattr(root, "right", None) is None:
        if not bits:
            symbols = [root.sym]
        else:
            symbols = [root.sym] * len(bits)
    else:
        symbols = []
        node = root
        for b in bits:
            node = node.left if b == '0' else node.right
            if node is None:
                break
            if node.sym is not None:
                symbols.append(node.sym)
                node = root
                if symbols[-1] == ('EOB',):
                    break

    coeffs = _rle_decode(symbols)
    coeffs = _apply_dc_dpcm_decode(coeffs)
    block8 = _inverse_zigzag(coeffs)
    return block8


# -------------------------------------------------------------------------
# Method 2: Raw (無壓縮，僅 Zigzag + 固定位元長度模擬)
# -------------------------------------------------------------------------

def encode_block_raw(block):
    """
    不壓縮模式：
    - Zigzag 掃描
    - 每個係數直接存 (模擬為 12 bits)
    """
    block = np.asarray(block)
    zz = _zigzag_flatten(block)
    
    # 模擬 bitstream：每個係數 12 bits (假設係數範圍 -2048~2047)
    # 這裡只為了計算大小，內容不重要，用 '0' 填充
    # 真實儲存會存 zz list
    bits_sim = '0' * (len(zz) * 12) 
    
    return {
        'codec': 'raw',
        'data': zz,
        'bits': bits_sim
    }

def decode_block_raw(stream_data):
    if not isinstance(stream_data, dict) or stream_data.get('codec') != 'raw':
        return stream_data # Should not happen if matched correctly
    
    zz = stream_data['data']
    block8 = _inverse_zigzag(zz)
    return block8


# -------------------------------------------------------------------------
# Method 3: RLE Only (無 Huffman，僅 RLE)
# -------------------------------------------------------------------------

def encode_block_rle(block):
    """
    RLE 模式 (無 Huffman)：
    - Zigzag -> RLE
    - 模擬編碼：Run(4bits) + Value(12bits) = 16 bits per symbol
    - EOB = 4 bits
    """
    block = np.asarray(block)
    zz = _zigzag_flatten(block)
    symbols = _rle_encode(zz)
    
    # 計算模擬的 bit 數
    total_len = 0
    for s in symbols:
        if s == ('EOB',):
            total_len += 4
        else:
            # (run, val) -> 4 bits run + 12 bits val
            total_len += (4 + 12)
            
    bits_sim = '0' * total_len
    
    return {
        'codec': 'rle',
        'symbols': symbols,
        'bits': bits_sim
    }

def decode_block_rle(stream_data):
    if not isinstance(stream_data, dict) or stream_data.get('codec') != 'rle':
        return stream_data
        
    symbols = stream_data['symbols']
    coeffs = _rle_decode(symbols)
    block8 = _inverse_zigzag(coeffs)
    return block8

# -------------------------------------------------------------------------
# Default / Baseline Aliases
# -------------------------------------------------------------------------
# 讓 main.py 的 baseline 能繼續運作 (對應到 huff)
encode_block = encode_block_huff
decode_block = decode_block_huff
