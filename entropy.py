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
_GLOBAL_HUFF_TABLE_BITS = 0


# JPEG-like DC 差分用的全域狀態
_DC_PREDICTORS = {"Y": 0, "Cb": 0, "Cr": 0}
_CURRENT_COMPONENT = "Y"  # 由 main.py 在處理 Y/Cb/Cr 前設定

# Shared DC/AC Huffman (JPEG-like DC/AC 分表，共用碼表；改為 per-component)
# dict[comp_name] -> codes / trees
_SHARED_DC_CODES_MAP = {}
_SHARED_DC_TREES_MAP = {}
_SHARED_AC_CODES_MAP = {}
_SHARED_AC_TREES_MAP = {}
_SHARED_DC_TABLE_BITS_MAP = {}
_SHARED_AC_TABLE_BITS_MAP = {}

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

def _estimate_table_bits_from_codes(codes: dict) -> int:
    """
    粗估 Huffman 表的 signaling 開銷（不追求位元精確，只求公平比較）：
    - 以 canonical 觀念，每個符號存 code length (~5 bits) + 符號標識 (~8 bits)
    - 預留一點 header，最低 16 bits
    """
    return max(16, len(codes) * 13)

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


# --- Bit helpers for JPEG-like DC/AC coding ---------------------------------


def _bit_length_for_value(v: int) -> int:
    """Number of bits to represent |v| in JPEG magnitude coding."""
    if v == 0:
        return 0
    return int(np.floor(np.log2(abs(v)))) + 1


def _value_to_bits(v: int, size: int) -> str:
    """
    JPEG magnitude bits for given value and category size.
    Positive: binary of v
    Negative: (1<<size) - 1 + v
    """
    if size == 0:
        return ""
    if v >= 0:
        return f"{v & ((1<<size)-1):0{size}b}"
    # negative
    return f"{((1<<size) - 1 + v) & ((1<<size)-1):0{size}b}"


def _receive_extend_bits(bitreader, size: int) -> int:
    """
    依 JPEG 的 receive/extend 邏輯還原帶符號值。
    bitreader: 提供 read(n) 回傳 int 的物件
    """
    if size == 0:
        return 0
    v = bitreader.read(size)
    if v < (1 << (size - 1)):
        v -= (1 << size) - 1
    return v


class _BitStrReader:
    """簡單的 bitreader，從字串 (e.g., '01011') 依序讀取 bits。"""
    def __init__(self, bitstr: str):
        self.bitstr = bitstr
        self.pos = 0

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > len(self.bitstr):
            raise EOFError("Not enough bits to read")
        val = int(self.bitstr[self.pos:self.pos+n], 2)
        self.pos += n
        return val


def prepare_huff_global(blocks):
    """
    根據整張圖（所有量化後 8x8 block）的 RLE 符號建立一棵「全圖共用」的 Huffman 樹。
    會將結果儲存在模組全域變數 _GLOBAL_HUFF_ROOT / _GLOBAL_HUFF_CODES 中。
    """
    global _GLOBAL_HUFF_ROOT, _GLOBAL_HUFF_CODES, _GLOBAL_HUFF_TABLE_BITS

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
    _GLOBAL_HUFF_TABLE_BITS = _estimate_table_bits_from_codes(codes)


def _build_prefix_tree_from_codes(codes: dict):
    """
    將 sym -> bitstr 的 dict 轉為 prefix tree，便於解碼。
    """
    tree = {}
    for sym, bitstr in codes.items():
        bitstr = str(bitstr)
        if bitstr == "":
            # 單節點退化情況：直接標上 sym
            tree["sym"] = sym
            continue
        node = tree
        for b in bitstr:
            node = node.setdefault(b, {})
        node["sym"] = sym
    return tree


def prepare_huff_dcac_shared(blocks_by_comp: dict):
    """
    建立 JPEG-like 的「共用 DC/AC Huffman 表」（改為 per-component）：
    - blocks_by_comp: dict{name->[quantized_blocks]}，例如 {'Y': [...], 'Cb': [...], 'Cr': [...]}
    - DC 使用 DPCM，AC 使用 (run,size) 符號；ZRL=0xF0, EOB=0x00
    產生全域 per-component 的 codes/trees
    """
    global _SHARED_DC_CODES_MAP, _SHARED_DC_TREES_MAP, _SHARED_AC_CODES_MAP, _SHARED_AC_TREES_MAP
    global _SHARED_DC_TABLE_BITS_MAP, _SHARED_AC_TABLE_BITS_MAP
    _SHARED_DC_CODES_MAP = {}
    _SHARED_DC_TREES_MAP = {}
    _SHARED_AC_CODES_MAP = {}
    _SHARED_AC_TREES_MAP = {}
    _SHARED_DC_TABLE_BITS_MAP = {}
    _SHARED_AC_TABLE_BITS_MAP = {}

    for comp_name, comp_blocks in blocks_by_comp.items():
        dc_freq = {}
        ac_freq = {}
        dc_pred = 0
        for block in comp_blocks:
            zz = _zigzag_flatten(block)
            # DC
            dc = int(zz[0])
            diff = dc - dc_pred
            dc_pred = dc
            size = _bit_length_for_value(diff)
            dc_freq[size] = dc_freq.get(size, 0) + 1

            # AC
            run = 0
            last_nz = 0
            for i in range(63, 0, -1):
                if zz[i] != 0:
                    last_nz = i
                    break
            for idx in range(1, last_nz + 1):
                v = int(zz[idx])
                if v == 0:
                    run += 1
                    if run == 16:
                        ac_freq[0xF0] = ac_freq.get(0xF0, 0) + 1  # ZRL
                        run = 0
                else:
                    size_v = _bit_length_for_value(v)
                    sym = (run << 4) | size_v
                    ac_freq[sym] = ac_freq.get(sym, 0) + 1
                    run = 0
            ac_freq[0x00] = ac_freq.get(0x00, 0) + 1  # EOB

        if not dc_freq:
            dc_freq = {0: 1}
        if not ac_freq:
            ac_freq = {0x00: 1}

        dc_root = _build_huffman(dc_freq)
        ac_root = _build_huffman(ac_freq)
        dc_codes = _gen_codes(dc_root)
        ac_codes = _gen_codes(ac_root)

        _SHARED_DC_CODES_MAP[comp_name] = dc_codes
        _SHARED_AC_CODES_MAP[comp_name] = ac_codes
        _SHARED_DC_TREES_MAP[comp_name] = _build_prefix_tree_from_codes(dc_codes)
        _SHARED_AC_TREES_MAP[comp_name] = _build_prefix_tree_from_codes(ac_codes)
        _SHARED_DC_TABLE_BITS_MAP[comp_name] = _estimate_table_bits_from_codes(dc_codes)
        _SHARED_AC_TABLE_BITS_MAP[comp_name] = _estimate_table_bits_from_codes(ac_codes)

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
        'table_bits': _estimate_table_bits_from_codes(codes),
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
                if symbols and symbols[-1] == ('EOB',):
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
        # 表的成本由 prepare_huff_global 記錄，全圖只算一次；在 main 統一加上
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
        'table_bits': _estimate_table_bits_from_codes(codes),
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
# Method 6: JPEG-like DC/AC shared Huffman (DC DPCM + AC (run,size) + 共用表)
# -------------------------------------------------------------------------


def encode_block_huff_dcac_shared(block):
    """
    使用共用 DC/AC Huffman 表（需先呼叫 prepare_huff_dcac_shared）。
    - DC: DPCM 後以「size」做 Huffman，接 magnitude bits。
    - AC: 符號 = (run,size)，ZRL=0xF0，EOB=0x00，接 magnitude bits。
    bitstream 以字串保存，用於統計長度；同時回傳樹以便解碼。
    """
    global _SHARED_DC_CODES, _SHARED_AC_CODES, _CURRENT_COMPONENT, _DC_PREDICTORS

    dc_codes = _SHARED_DC_CODES_MAP.get(_CURRENT_COMPONENT)
    ac_codes = _SHARED_AC_CODES_MAP.get(_CURRENT_COMPONENT)
    if dc_codes is None or ac_codes is None:
        # 尚未準備好共用表，退回 per-block Huffman
        return encode_block_huff(block)

    block = np.asarray(block)
    zz = _zigzag_flatten(block)

    # DC
    dc = int(zz[0])
    pred = _DC_PREDICTORS.get(_CURRENT_COMPONENT, 0)
    diff = dc - pred
    _DC_PREDICTORS[_CURRENT_COMPONENT] = dc
    size_dc = _bit_length_for_value(diff)
    dc_bits = _value_to_bits(diff, size_dc)
    dc_code = dc_codes.get(size_dc, '')

    # AC
    ac_bitstr = ""
    run = 0
    last_nz = 0
    for i in range(63, 0, -1):
        if zz[i] != 0:
            last_nz = i
            break
    for idx in range(1, last_nz + 1):
        v = int(zz[idx])
        if v == 0:
            run += 1
            if run == 16:
                # ZRL
                ac_bitstr += ac_codes.get(0xF0, '')
                run = 0
        else:
            size_v = _bit_length_for_value(v)
            sym = (run << 4) | size_v
            ac_bitstr += ac_codes.get(sym, '')
            ac_bitstr += _value_to_bits(v, size_v)
            run = 0
    # EOB
    ac_bitstr += ac_codes.get(0x00, '')

    bits = dc_code + dc_bits + ac_bitstr
    return {
        'codec': 'huff_dcac_shared',
        'bits': bits,
        # 表成本在 prepare_huff_dcac_shared 記錄，全圖每分量只算一次；main 統一加上
    }


def decode_block_huff_dcac_shared(stream_data):
    """
    解碼共用 DC/AC Huffman 表的 bitstream。
    - 需先呼叫 prepare_huff_dcac_shared 建立全域樹。
    - 使用 _CURRENT_COMPONENT 的 DC predictor。
    """
    global _SHARED_DC_TREE, _SHARED_AC_TREE, _DC_PREDICTORS, _CURRENT_COMPONENT

    if not isinstance(stream_data, dict) or stream_data.get('codec') != 'huff_dcac_shared':
        return stream_data

    dc_tree = _SHARED_DC_TREES_MAP.get(_CURRENT_COMPONENT)
    ac_tree = _SHARED_AC_TREES_MAP.get(_CURRENT_COMPONENT)
    if dc_tree is None or ac_tree is None:
        return stream_data

    bitreader = _BitStrReader(stream_data['bits'])

    # DC decode
    node = dc_tree
    while "sym" not in node:
        b = bitreader.read(1)
        node = node.get("1" if b else "0")
        if node is None:
            raise ValueError("Invalid DC Huffman code")
    size_dc = node["sym"]
    diff = _receive_extend_bits(bitreader, size_dc)
    pred = _DC_PREDICTORS.get(_CURRENT_COMPONENT, 0)
    dc = pred + diff
    _DC_PREDICTORS[_CURRENT_COMPONENT] = dc

    coeffs = [0] * 64
    coeffs[0] = dc

    # AC decode
    idx = 1
    while idx < 64:
        node = ac_tree
        while "sym" not in node:
            b = bitreader.read(1)
            node = node.get("1" if b else "0")
            if node is None:
                raise ValueError("Invalid AC Huffman code")
        sym = node["sym"]
        if sym == 0x00:  # EOB
            break
        if sym == 0xF0:  # ZRL
            idx += 16
            continue
        run = (sym >> 4) & 0x0F
        size = sym & 0x0F
        idx += run
        if idx >= 64:
            break
        val = _receive_extend_bits(bitreader, size)
        coeffs[idx] = val
        idx += 1

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


# -------------------------------------------------------------------------
# Signaling cost helpers (for main to add table overhead)
# -------------------------------------------------------------------------

def get_global_huff_table_bits():
    """回傳全圖共用 Huffman 的表成本（bits）。"""
    return _GLOBAL_HUFF_TABLE_BITS


def get_shared_dcac_table_bits_total():
    """
    回傳 JPEG-like DC/AC 共用表的總 signaling bits（DC+AC，三個分量合計）。
    """
    total = 0
    for comp in _SHARED_DC_TABLE_BITS_MAP:
        total += _SHARED_DC_TABLE_BITS_MAP.get(comp, 0)
    for comp in _SHARED_AC_TABLE_BITS_MAP:
        total += _SHARED_AC_TABLE_BITS_MAP.get(comp, 0)
    return total
