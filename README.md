## JPEG Encoder/Decoder 專案說明

這是一個簡化版的 JPEG Encoder/Decoder 專案，用來實作與比較不同的 JPEG 元件（顏色轉換、子採樣、DCT/IDCT、量化與熵編碼等）。

主要進入點為 `main.py`，可以針對單張圖片或整個資料夾進行編碼與解碼，並輸出重建影像與統計結果。

---

## 環境與依賴

- **Python**: 建議 3.8 以上

```bash
pip install -r requirements.txt
```

---

## 專案結構（重點檔案）

- `main.py`：專案主程式（命令列介面，負責讀檔、呼叫 JPEG pipeline、存結果）
- `transforms.py`：DCT / IDCT（成員 C）
- `sampler.py`：色度降採樣 / 升採樣（成員 B）
- `entropy.py`：量化與熵編碼（成員 A）
- `config.py`：JPEG 量化表（亮度與色度）
- `utils.py`：工具函式（色彩空間轉換、切 block、重建、PSNR 計算等）
- `images/`：輸入圖片資料夾（例如 Kodak PNG）
- `outputs/`：結果輸出資料夾（程式執行時自動建立）

---

## `main.py` 使用方式

### 基本命令列介面

在專案根目錄執行：

```bash
python main.py --image_dir images --output_dir outputs
```

- `--image_dir`：輸入圖片資料夾（預設為 `images`）
- `--output_dir`：輸出資料夾（預設為 `outputs`，程式會在其下依照輸入資料夾名稱再建立子目錄）

若 `image_dir` 底下有多張圖片，程式會自動逐張處理。

### 處理單張圖片

若只想處理某一張圖片，可以指定 `--image_name`：

```bash
python main.py \
  --image_dir images \
  --image_name 1.png \
  --output_dir outputs
```

- `--image_name`：圖片檔名（例如 `1.png`）；若未指定，會處理資料夾中所有支援格式 (`.jpg`, `.jpeg`, `.png`, `.bmp`)。

程式會在輸出資料夾中產生：

- 對比圖：`result_<原檔名去副檔名>.png`（左半為原圖，右半為重建圖）
- 統計 JSON：`result_<原檔名去副檔名>.json`，內容包含：
  - `psnr`
  - `stats.original_bytes`
  - `stats.compressed_bytes`
  - `stats.compression_ratio`
  - `stats.time_encoding`、`stats.time_decoding_dct`、`stats.time_upsample`、`stats.total_time` 等

---

## 方法選擇參數

`main.py` 提供多種方法的後綴切換機制，透過命令列參數指定：

```bash
python main.py \
  --image_dir images \
  --output_dir outputs \
  --dct_method baseline \
  --idct_method baseline \
  --downsample_method baseline \
  --upsample_method baseline \
  --entropy_method baseline
```

### DCT / IDCT

- `--dct_method`：DCT 方法（預設 `baseline`，對應 `transforms.forward_dct_block`）
- `--idct_method`：IDCT 方法（預設 `baseline`，對應 `transforms.inverse_dct_block`）

若你實作了新的方法，例如 `forward_dct_block_fft`，可以這樣呼叫：

```bash
python main.py --dct_method fft
```

內部會嘗試呼叫 `transforms.forward_dct_block_fft`。

### 降採樣 / 升採樣

- `--downsample_method`：色度降採樣方法（預設 `baseline`，對應 `sampler.downsample`）
- `--upsample_method`：色度升採樣方法（預設 `baseline`，對應 `sampler.upsample`）

例如若成員 B 實作了 `upsample_bilinear`，可以這樣呼叫：

```bash
python main.py --upsample_method bilinear
```

內部會呼叫 `sampler.upsample_bilinear`。

### 熵編碼（Entropy Coding）

- `--entropy_method`：熵編碼方法（預設 `baseline`）

目前在 `entropy.py` 中已提供幾種版本：

- `baseline`：對應到 `encode_block` / `decode_block`，預設實作為 Huffman + RLE（`encode_block_huff`）。
- `raw`：`encode_block_raw` / `decode_block_raw`，僅做 Zigzag 掃描與固定長度 bit 數估算，**不做真正壓縮**（可當作 baseline 比較）。
- `rle`：`encode_block_rle` / `decode_block_rle`，只做零值的 Run-Length Encoding（RLE），不使用 Huffman。
- `huff`：`encode_block_huff` / `decode_block_huff`，完整的 Zigzag + RLE + Huffman。

使用範例：

```bash
# 無熵壓縮（raw）
python main.py --image_dir images --output_dir outputs_raw --entropy_method raw

# 只使用 RLE
python main.py --image_dir images --output_dir outputs_rle --entropy_method rle

# Huffman + RLE（與 baseline 類似）
python main.py --image_dir images --output_dir outputs_huff --entropy_method huff
```

> 註：本說明僅介紹 `main.py` 的一般使用方式，額外的實驗腳本與分析工具不在此文件中說明。
