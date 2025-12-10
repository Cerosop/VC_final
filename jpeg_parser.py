import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config
import entropy
import transforms
import sampler
import utils


# ----------------------------------------------------------------------
# 基本資料結構
# ----------------------------------------------------------------------


@dataclass
class QuantTable:
    id: int
    table: np.ndarray  # 8x8


@dataclass
class HuffmanTable:
    tc: int  # 0 = DC, 1 = AC
    th: int  # table id
    # dictionary: bitstring (str) -> symbol (int)
    codes: Dict[str, int]
    # 解碼用 prefix tree: dict -> (dict or symbol)
    tree: dict


@dataclass
class ComponentSpec:
    id: int
    h: int  # 水平 sampling factor
    v: int  # 垂直 sampling factor
    q_id: int  # 對應的量化表 id


@dataclass
class FrameInfo:
    precision: int
    height: int
    width: int
    components: List[ComponentSpec]
    max_h: int
    max_v: int


@dataclass
class ScanComponentSpec:
    id: int
    dc_table_id: int
    ac_table_id: int


@dataclass
class ScanInfo:
    components: List[ScanComponentSpec]


class BitReader:
    """簡單的 bitstream 讀取器，處理 JPEG byte stuffing (0xFF00)。"""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0  # byte index
        self.bit_buf = 0
        self.bit_count = 0

    def _read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise EOFError("Bitstream exhausted")
        b = self.data[self.pos]
        self.pos += 1
        if b == 0xFF:
            if self.pos >= len(self.data):
                raise EOFError("Unexpected EOF after 0xFF")
            nb = self.data[self.pos]
            # Stuffed byte 0x00: real data 0xFF
            if nb == 0x00:
                self.pos += 1
                return 0xFF
            # Restart markers (FFD0..FFD7) — not supported, abort early
            if 0xD0 <= nb <= 0xD7:
                raise NotImplementedError("Restart markers (FFD0-FFD7) are not supported yet.")
            # EOI inside entropy: treat as end of stream
            if nb == 0xD9:
                raise EOFError("Encountered EOI inside entropy data")
            # Any other marker inside entropy is unexpected
            raise ValueError(f"Unexpected marker FF{nb:02X} inside entropy data")
        return b

    def _ensure_bits(self, n: int):
        while self.bit_count < n:
            b = self._read_byte()
            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_count += 8

    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        self._ensure_bits(n)
        mask = (1 << n) - 1
        val = (self.bit_buf >> (self.bit_count - n)) & mask
        self.bit_count -= n
        # 避免 bit_buf 無限膨脹：只保留尚未讀取的低位 bits
        if self.bit_count == 0:
            self.bit_buf = 0
        else:
            self.bit_buf &= (1 << self.bit_count) - 1
        return val


# ----------------------------------------------------------------------
# Marker 解析
# ----------------------------------------------------------------------


def _read_u8(f) -> int:
    b = f.read(1)
    if not b:
        raise EOFError("Unexpected EOF while reading byte")
    return b[0]


def _read_u16(f) -> int:
    b = f.read(2)
    if len(b) != 2:
        raise EOFError("Unexpected EOF while reading u16")
    return struct.unpack(">H", b)[0]


def _read_marker(f) -> int:
    # 找到 0xFF 開頭，再讀下一個非 0xFF 的 byte 當 marker
    b = _read_u8(f)
    while b != 0xFF:
        b = _read_u8(f)
    # 跳過填充的 0xFF
    while True:
        m = _read_u8(f)
        if m != 0xFF:
            return m


def _parse_dqt(f, quant_tables: Dict[int, QuantTable]):
    length = _read_u16(f)
    bytes_left = length - 2
    while bytes_left > 0:
        pq_tq = _read_u8(f)
        bytes_left -= 1
        pq = (pq_tq >> 4) & 0x0F  # precision: 0 = 8-bit, 1 = 16-bit
        tq = pq_tq & 0x0F         # table id
        if pq != 0:
            raise NotImplementedError("Only 8-bit precision quant tables supported")
        q_vals = f.read(64)
        if len(q_vals) != 64:
            raise EOFError("Unexpected EOF in DQT")
        bytes_left -= 64
        q = np.frombuffer(q_vals, dtype=np.uint8).astype(np.int32).reshape((8, 8))
        quant_tables[tq] = QuantTable(id=tq, table=q)


def _build_huffman_codes(code_lengths: List[int], symbols: List[int]) -> Tuple[Dict[str, int], dict]:
    """根據 JPEG DHT 給的 code length 與符號順序建立 Huffman codeword 對應，並建 prefix tree 方便快速解碼。"""
    codes: Dict[str, int] = {}
    tree: dict = {}
    code = 0
    idx = 0
    for bit_len, count in enumerate(code_lengths, start=1):
        for _ in range(count):
            sym = symbols[idx]
            idx += 1
            bitstr = f"{code:0{bit_len}b}"
            codes[bitstr] = sym
            # 建立樹
            node = tree
            for b in bitstr:
                if b not in node:
                    node[b] = {}
                node = node[b]
            node["sym"] = sym
            code += 1
        code <<= 1
    return codes, tree


def _parse_dht(f, huff_tables: Dict[Tuple[int, int], HuffmanTable]):
    length = _read_u16(f)
    bytes_left = length - 2
    while bytes_left > 0:
        tc_th = _read_u8(f)
        bytes_left -= 1
        tc = (tc_th >> 4) & 0x0F  # 0=DC, 1=AC
        th = tc_th & 0x0F
        # 讀每種長度的 code 個數 (1..16)
        counts_bytes = f.read(16)
        if len(counts_bytes) != 16:
            raise EOFError("Unexpected EOF in DHT")
        bytes_left -= 16
        code_lengths = [b for b in counts_bytes]
        total_syms = sum(code_lengths)
        sym_bytes = f.read(total_syms)
        if len(sym_bytes) != total_syms:
            raise EOFError("Unexpected EOF in DHT symbols")
        bytes_left -= total_syms
        symbols = [b for b in sym_bytes]
        codes, tree = _build_huffman_codes(code_lengths, symbols)
        huff_tables[(tc, th)] = HuffmanTable(tc=tc, th=th, codes=codes, tree=tree)


def _parse_sof0(f) -> FrameInfo:
    length = _read_u16(f)
    precision = _read_u8(f)
    height = _read_u16(f)
    width = _read_u16(f)
    n_comp = _read_u8(f)
    components: List[ComponentSpec] = []
    max_h = 0
    max_v = 0
    for _ in range(n_comp):
        cid = _read_u8(f)
        hv = _read_u8(f)
        q_id = _read_u8(f)
        h = (hv >> 4) & 0x0F
        v = hv & 0x0F
        components.append(ComponentSpec(id=cid, h=h, v=v, q_id=q_id))
        max_h = max(max_h, h)
        max_v = max(max_v, v)

    # 跳過剩餘 bytes（理論上 length 剛好用完，但保險起見）
    remaining = length - (8 + 3 * n_comp)
    if remaining > 0:
        f.read(remaining)

    return FrameInfo(
        precision=precision,
        height=height,
        width=width,
        components=components,
        max_h=max_h,
        max_v=max_v,
    )


def _parse_sos(f) -> ScanInfo:
    length = _read_u16(f)
    n_comp = _read_u8(f)
    comps: List[ScanComponentSpec] = []
    for _ in range(n_comp):
        cid = _read_u8(f)
        td_ta = _read_u8(f)
        td = (td_ta >> 4) & 0x0F  # DC table id
        ta = td_ta & 0x0F         # AC table id
        comps.append(ScanComponentSpec(id=cid, dc_table_id=td, ac_table_id=ta))

    # 忽略 spectral selection 與 approximation（只支援 baseline）
    f.read(3)
    # 剩餘 bytes 理論上為 0
    remaining = length - (6 + 2 * n_comp)
    if remaining > 0:
        f.read(remaining)

    return ScanInfo(components=comps)


# ----------------------------------------------------------------------
# 熵解碼（僅支援 baseline, non-progressive, 無 restart）
# ----------------------------------------------------------------------


def _receive_extend(br: BitReader, size: int) -> int:
    """JPEG 標準中的 receive+extend，用於還原 DC/AC 真實值。"""
    if size == 0:
        return 0
    v = br.read_bits(size)
    # 若最高位為 0，則為負數，要做 sign extension
    if v < (1 << (size - 1)):
        v -= (1 << size) - 1
    return v


def _huff_decode_symbol(br: BitReader, table: HuffmanTable) -> int:
    node = table.tree
    depth = 0
    while True:
        bit = br.read_bits(1)
        depth += 1
        node = node.get("1" if bit else "0")
        if node is None:
            raise ValueError("Invalid Huffman code encountered")
        if "sym" in node:
            return node["sym"]
        if depth > 16:
            raise ValueError("Invalid Huffman code encountered (too long)")


def _decode_mcu_blocks(
    br: BitReader,
    frame: FrameInfo,
    scan: ScanInfo,
    quant_tables: Dict[int, QuantTable],
    huff_tables: Dict[Tuple[int, int], HuffmanTable],
):
    """
    解碼所有 MCU，回傳各 component 的 8x8 block 陣列：
    dict[component_id] -> List[np.ndarray (8x8)]
    僅支援 baseline、單一 scan、無 restart markers。
    """
    n_comp = len(frame.components)
    # 建立 id -> ComponentSpec / ScanSpec 快速查表
    comp_by_id = {c.id: c for c in frame.components}
    scan_by_id = {c.id: c for c in scan.components}

    # 目前只支援 3 component (Y/Cb/Cr)
    if n_comp != 3:
        raise NotImplementedError("Only 3-component baseline JPEG is supported.")

    # MCU 數量
    mcu_width = (frame.width + frame.max_h * 8 - 1) // (frame.max_h * 8)
    mcu_height = (frame.height + frame.max_v * 8 - 1) // (frame.max_v * 8)

    # DC predictor per component id
    dc_pred = {c.id: 0 for c in frame.components}

    # 儲存每個 component 的 block 順序
    blocks: Dict[int, List[np.ndarray]] = {c.id: [] for c in frame.components}

    for _my in range(mcu_height):
        for _mx in range(mcu_width):
            # 每個 MCU 依照 scan 順序處理各 component
            for sc in scan.components:
                cspec = comp_by_id[sc.id]
                qtbl = quant_tables[cspec.q_id].table
                h_dc = huff_tables[(0, sc.dc_table_id)]
                h_ac = huff_tables[(1, sc.ac_table_id)]

                # 該 component 在 MCU 內的 block 數量 = h * v
                for _ in range(cspec.h * cspec.v):
                    # DC
                    size = _huff_decode_symbol(br, h_dc)
                    diff = _receive_extend(br, size)
                    dc = dc_pred[cspec.id] + diff
                    dc_pred[cspec.id] = dc

                    coeffs = [0] * 64
                    coeffs[0] = dc

                    # AC
                    idx = 1
                    while idx < 64:
                        sym = _huff_decode_symbol(br, h_ac)
                        if sym == 0x00:  # EOB
                            break
                        if sym == 0xF0:  # ZRL: 16 個連續 0
                            idx += 16
                            continue
                        run = (sym >> 4) & 0x0F
                        size = sym & 0x0F
                        idx += run
                        if idx >= 64:
                            break
                        val = _receive_extend(br, size)
                        coeffs[idx] = val
                        idx += 1

                    # 反 Zigzag 成 8x8 block，暫存「量化前」係數（之後再做 dequant+IDCT）
                    block8 = np.zeros((8, 8), dtype=np.int32)
                    for (r, c), v in zip(entropy._ZZ_IDX, coeffs):
                        block8[r, c] = v

                    # 立即 dequant，與現有 pipeline 一致
                    block8 = block8 * qtbl
                    blocks[cspec.id].append(block8.astype(np.float64))

    return blocks, mcu_width, mcu_height


# ----------------------------------------------------------------------
# 對外介面：decode_baseline_jpeg
# ----------------------------------------------------------------------


def decode_baseline_jpeg(path: str | Path) -> np.ndarray:
    """
    讀取 baseline JPEG 檔案並 decode 成 RGB 影像。
    僅支援：
      - baseline DCT (SOF0)
      - non-progressive
      - 3 components (Y/Cb/Cr)
      - 無 restart markers

    回傳：np.ndarray, shape=(H, W, 3), dtype=np.uint8
    """
    path = Path(path)
    with path.open("rb") as f:
        # 檢查 SOI
        if _read_u8(f) != 0xFF or _read_u8(f) != 0xD8:
            raise ValueError("Not a JPEG file (missing SOI)")

        quant_tables: Dict[int, QuantTable] = {}
        huff_tables: Dict[Tuple[int, int], HuffmanTable] = {}
        frame: FrameInfo | None = None
        scan: ScanInfo | None = None
        entropy_data: bytes | None = None

        while True:
            marker = _read_marker(f)
            if marker == 0xD9:  # EOI
                break
            elif marker == 0xDB:  # DQT
                _parse_dqt(f, quant_tables)
            elif marker == 0xC4:  # DHT
                _parse_dht(f, huff_tables)
            elif marker == 0xC0:  # SOF0 (baseline)
                frame = _parse_sof0(f)
            elif marker == 0xDA:  # SOS
                scan = _parse_sos(f)
                # 讀到檔案尾，並嘗試截掉最後的 EOI (FFD9) / 填充 marker
                raw_tail = f.read()
                # 找到最後一個 0xFFD9，截掉後面的 marker
                cut = raw_tail.rfind(b"\xFF\xD9")
                if cut != -1:
                    raw_tail = raw_tail[:cut]
                entropy_data = raw_tail
                break
            else:
                # 其他 marker：讀長度並跳過內容
                length = _read_u16(f)
                f.read(length - 2)

    if frame is None or scan is None or entropy_data is None:
        raise ValueError("Incomplete JPEG: missing SOF0/SOS/entropy data")

    if frame.precision != 8:
        raise NotImplementedError("Only 8-bit precision baseline JPEG is supported.")

    # 目前只支援 3 components
    if len(frame.components) != 3:
        raise NotImplementedError("Only 3-component JPEG is supported.")

    # 熵解碼
    br = BitReader(entropy_data)
    # 注意：這裡僅作 baseline 解碼，未支援 restart markers
    blocks_by_id, mcu_w, mcu_h = _decode_mcu_blocks(
        br, frame, scan, quant_tables, huff_tables
    )

    # 將 block 陣列轉成 Y/Cb/Cr 平面，僅支援最常見的 4:2:0 或 4:4:4
    comp_by_id = {c.id: c for c in frame.components}
    # 嘗試推斷 Y/Cb/Cr 的 component id（通常為 1,2,3）
    ids_sorted = sorted(comp_by_id.keys())
    cY, cCb, cCr = ids_sorted

    cy = comp_by_id[cY]
    ccb = comp_by_id[cCb]
    ccr = comp_by_id[cCr]

    H_max, V_max = frame.max_h, frame.max_v

    def _reconstruct_plane(blocks_list: List[np.ndarray], cspec: ComponentSpec) -> np.ndarray:
        """
        依照 MCU 掃描順序，把 blocks 以 (mx,my) 為外圈、(h,v) 為內圈放回平面。
        並在 IDCT 後加上 128 做 level shift（JPEG 標準反量化後的 IDCT 輸出是以 0 為中心）。
        """
        plane_h = mcu_h * cspec.v * 8
        plane_w = mcu_w * cspec.h * 8
        plane = np.zeros((plane_h, plane_w), dtype=np.float64)

        idx = 0
        for my in range(mcu_h):
            for mx in range(mcu_w):
                for b_v in range(cspec.v):
                    for b_h in range(cspec.h):
                        if idx >= len(blocks_list):
                            break
                        block = blocks_list[idx]
                        idx += 1

                        spatial_block = transforms.inverse_dct_block(block)
                        spatial_block += 128.0  # level shift

                        y_off = (my * cspec.v + b_v) * 8
                        x_off = (mx * cspec.h + b_h) * 8
                        if y_off + 8 <= plane_h and x_off + 8 <= plane_w:
                            plane[y_off : y_off + 8, x_off : x_off + 8] = spatial_block
        return plane

    Y_plane = _reconstruct_plane(blocks_by_id[cY], cy)
    Cb_plane = _reconstruct_plane(blocks_by_id[cCb], ccb)
    Cr_plane = _reconstruct_plane(blocks_by_id[cCr], ccr)

    # 若有 subsampling，將 Cb/Cr 升樣到影像大小
    H, W = frame.height, frame.width
    if cy.h == H_max and cy.v == V_max and ccb.h == 1 and ccb.v == 1:
        # 常見情況：Y (2x2), Cb/Cr (1x1) 4:2:0 or 4:2:2，這裡簡化使用最近鄰升樣
        Cb_plane = sampler.upsample(Cb_plane, Y_plane.shape[0], Y_plane.shape[1])
        Cr_plane = sampler.upsample(Cr_plane, Y_plane.shape[0], Y_plane.shape[1])

    # 裁切到精確寬高
    Y_plane = Y_plane[:H, :W]
    Cb_plane = Cb_plane[:H, :W]
    Cr_plane = Cr_plane[:H, :W]

    ycbcr = np.stack([Y_plane, Cb_plane, Cr_plane], axis=2)
    rgb = utils.ycbcr_to_rgb(ycbcr)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


__all__ = ["decode_baseline_jpeg"]


if __name__ == "__main__":
    """
    簡易 CLI：
    從專案根目錄執行本檔，會自動將 baseline_jpegs/ 底下所有 .jpg/.jpeg
    解碼成 PNG，輸出到 baseline_jpegs_decoded/。

    範例：
        python jpeg_parser.py
    """
    import os
    from PIL import Image

    project_root = Path(__file__).parent
    input_dir = project_root / "baseline_jpegs"
    output_dir = project_root / "baseline_jpegs_decoded"
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    if not input_dir.exists():
        print(f"[Warn] baseline_jpegs 資料夾不存在：{input_dir}")
    else:
        files = [p for p in input_dir.iterdir() if p.suffix in exts]
        if not files:
            print(f"[Info] baseline_jpegs 內沒有 .jpg/.jpeg 檔案。")
        else:
            print(f"[Info] 找到 {len(files)} 個 baseline JPEG 檔案，開始解碼...")
            for p in files:
                try:
                    rgb = decode_baseline_jpeg(p)
                    out_path = output_dir / f"{p.stem}.png"
                    Image.fromarray(rgb).save(out_path)
                    print(f"  [OK] {p.name} -> {out_path.name}")
                except Exception as e:
                    print(f"  [Fail] {p.name}: {e}")
