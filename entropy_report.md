# Entropy 實驗報告

本報告說明我們在 JPEG-like pipeline 中，針對不同熵編碼方法與量化品質的實驗結果。

## 背景與目的

- 實驗包括:
  1. 不同熵編碼策略（raw / rle / huff / huff_global / huff_dpcm）對壓縮率、bpp、PSNR、編碼時間的影響。
  2. 不同 JPEG 品質 (Quality) 下的畫質與碼率取捨（RD 曲線）。
- 資料集：Kodak 24 張自然圖片 + 1 張 Lena，共 25 張，已裁成 16 的倍數。

---

## 方法簡介（六種熵編碼）

| 方法             | 做法                                                                                                     | 直覺比喻                            |
| ---------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| raw              | 只做 Zigzag，沒有任何壓縮；每個係數假設用固定 12 bits                                                    | 把數字原封不動存起來                |
| rle              | Zigzag 後針對連續 0 做零值游程編碼 (Run-Length Encoding)                                                 | 把「很多 0」縮成「有幾個 0」        |
| huff             | 每個 8×8 block 自己統計符號，建一棵 Huffman 樹，再對該 block 編碼                                        | 針對「這個小區塊」的常見符號給短碼  |
| huff_global      | 全圖的符號統計共用一棵 Huffman 樹，再對每個 block 編碼                                                   | 先看完整張圖的統計，再用同一套碼表  |
| huff_dpcm        | 先對 DC 做差分 (DPCM) 讓 DC 更集中，再做 per-block Huffman                                               | 先把 DC 平滑化，再壓縮              |
| huff_dcac_shared | JPEG 風格：DC 做 DPCM，AC 用 (run,size) 符號；Y/Cb/Cr 各自共用一套 Huffman 表；數值幅度用 magnitude bits | 直接貼近 baseline JPEG 的熵編碼流程 |

---

## 實驗設定與指標

- 固定：DCT/IDCT baseline；4:2:0 baseline 子採樣；標準量化表；升樣 baseline。
- 熵方法：raw / rle / huff / huff_global / huff_dpcm。
- 指標：
  - PSNR (dB) — 畫質
  - Compression Ratio = 原始大小 / 壓縮後大小
  - bpp (bits per pixel)
  - time_encoding_ms（只含 DCT+量化+熵編碼）
- 圖檔：
  - 熵方法比較：`plots/entropy_summary.png`
  - 品質 RD 曲線：`plots/quality_rd.png`

---

## 實驗一：熵編碼方法比較（平均 25 張，含碼表 signaling 成本）

| 方法                             | PSNR (dB) | 壓縮率 (orig/comp) | bpp       | 編碼時間 (ms) |
| -------------------------------- | --------- | ------------------ | --------- | ------------- |
| raw                              | 33.67     | 1.33×              | 18.000    | 145.8         |
| rle                              | 33.67     | 9.74×              | 2.680     | 172.0         |
| huff                             | 33.67     | 10.43×             | 2.437     | 292.4         |
| huff_global                      | 33.67     | 25.70×             | 0.998     | 257.0         |
| huff_dpcm                        | 33.67     | 11.14×             | 2.300     | 257.6         |
| **huff_dcac_shared (JPEG-like)** | 33.67     | **30.96×**         | **0.848** | 404.8         |

![](plots/entropy_summary.png)

> 上：壓縮率與 bpp；下：編碼時間。

### 結論

1. **Huffman 仍是主要壓縮來源**：raw→rle→huff 的落差最大。
2. **碼表成本已計入**：per-block Huffman 不再虛高；共用表（huff_global / huff_dcac_shared）在「含表成本」下壓縮率明顯提升。
3. **JPEG 風格 (huff_dcac_shared)**：DC DPCM + AC (run,size) + per-component 共用表 + magnitude bits，最貼近 baseline JPEG；壓縮率 30.96×，bpp 0.848，但編碼時間較長。
4. **huff_global**：共享一張表、單層符號 (run,value)，比 per-block 更省 bits 與時間；當有大樣本統計時效果佳。
5. **per-block Huffman/huff_dpcm**：在含表成本後，壓縮率落在 ~10–11×，bpp ~2.3–2.4，時間中等。

---

## 實驗二：Quality 掃描（huff_dcac_shared，Q=30/50/70/90，平均 25 張）

| Quality | PSNR (dB) | 壓縮率 | bpp   |
| ------- | --------- | ------ | ----- |
| 30      | 32.72     | 44.63× | 0.593 |
| 50      | 29.07     | 5.86×  | 4.116 |
| 70      | 28.15     | 4.81×  | 4.988 |
| 90      | 27.79     | 3.45×  | 6.965 |

![](plots/quality_rd.png)

> PSNR vs bpp（Rate–Distortion 曲線；bpp=每像素實際位元數，最能代表碼率）

### 結論

- Quality 越高：PSNR 越好，但 bpp 也增加；曲線呈典型的 Rate–Distortion 取捨。
- Q30 壓得最兇（bpp ~0.38，PSNR ~32.7 dB）；Q90 畫質最好但 bpp ~5.66。

### 為什麼橫軸用 bpp？為什麼要做實驗二？

- bpp（bits per pixel）是最直接的碼率度量，和「每像素要花多少位元」一一對應；RD 曲線的橫軸慣用碼率（bpp 或 Mbps），縱軸用畫質（PSNR/SSIM）。
- JPEG 的 Quality 會改變量化表，進而改變係數稀疏度；稀疏度決定熵編碼的位元數，所以 Quality → 係數分布 → 熵碼長（bpp）形成一條 RD 曲線。
- 這個實驗展示「畫質 vs 碼率」的權衡，也讓我們量化在不同 Quality 下（這裡用 huff_dcac_shared）熵編碼實際輸出的位元數。

---

## Experiment Script Usage

```bash
# 熵方法掃描（產生 entropy_results.csv 與圖）
python experiments/run_entropy_experiments.py
python experiments/plot_entropy_results.py

# 品質掃描（產生 quality_results.csv 與 RD 圖）
python experiments/run_quality_experiments.py
python experiments/plot_quality_results.py

# 單張示範（生成 comparison_1.png）
python experiments/experiment_entropy.py images/1.png
```

---

## 這些結果的意義

- **RLE → Huffman → DC DPCM**：每一步都在「讓符號分布更集中」或「更懂統計」，壓縮率逐步上升。
- **全圖共用 vs 每 block**：共用碼表省時間；若計入碼表 signaling 成本，實務 JPEG 會偏向全圖/通道共用碼表。
- **RD 曲線**：呈現經典的「品質 vs 碼率」權衡，可類比視訊編碼中的 RD 分析。
