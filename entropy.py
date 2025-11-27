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
    out = np.zeros((8,8), dtype=np.int32)
    for (r,c), v in zip(_ZZ_IDX, coeffs):
        out[r, c] = v
    return out


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


def encode_block(block):
    """
    將量化後的 8x8 區塊轉為 Bitstream：
    - Zigzag -> RLE(零) -> 建立 Huffman Tree -> 逐 bit 編碼
    回傳：{"codec":"huff_rle_zigzag_v1","tree":root_node,"bits":str}
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
        'codec': 'huff_rle_zigzag_v1',
        'tree': root,
        'bits': bits,
    }

def decode_block(stream_data):
    """
    將 Bitstream 解碼回 8x8 量化區塊：
    - 逐 bit 讀取，沿 Huffman 樹遍歷至葉節點取得符號
    - 反 RLE -> 反 Zigzag
    """
    if isinstance(stream_data, np.ndarray):
        # 兼容舊格式：直接回傳
        return stream_data

    if not isinstance(stream_data, dict) or stream_data.get('codec') != 'huff_rle_zigzag_v1':
        # 無法辨識就原封不動回傳
        return stream_data

    root = stream_data['tree']
    bits = stream_data['bits']

    # bit-by-bit 遍歷
    symbols = []
    node = root
    for b in bits:
        node = node.left if b == '0' else node.right
        if node.sym is not None:
            symbols.append(node.sym)
            node = root
            # 提前終止：遇到 EOB 且已足夠
            if symbols[-1] == ('EOB',):
                break

    coeffs = _rle_decode(symbols)
    block8 = _inverse_zigzag(coeffs)
    return block8

# -----------------------