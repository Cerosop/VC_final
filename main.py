import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json

# 引入模組
import config
import utils
import transforms
import sampler
import entropy

def _resolve_func(module, base_name: str, method: str):
    """Return callable based on method suffix; fallback to baseline."""
    if method and method.lower() != 'baseline':
        name = f"{base_name}_{method}"
        if hasattr(module, name):
            return getattr(module, name)
    return getattr(module, base_name)


def jpeg_pipeline(img_arr, dct_method='baseline', idct_method='baseline',
                  downsample_method='baseline', upsample_method='baseline',
                  entropy_method='baseline'):
    """
    執行完整的 JPEG 編碼與解碼流程 (含效能監測)
    """
    h, w, _ = img_arr.shape
    stats = {} # 用來存統計數據
    
    t_start = time.time() # 開始計時

    # --- ENCODER ---
    # 1. Color Conversion
    ycbcr = utils.rgb_to_ycbcr(img_arr)
    Y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    # 2. Downsampling (成員 B)
    t0 = time.time()
    downsample_fn = _resolve_func(sampler, 'downsample', downsample_method)
    Cb_sub = downsample_fn(Cb)
    Cr_sub = downsample_fn(Cr)
    stats['time_downsample'] = (time.time() - t0) * 1000 # ms

    # 3. Block Splitting
    Y_blocks, h_y, w_y = utils.make_blocks(Y)
    Cb_blocks, h_c, w_c = utils.make_blocks(Cb_sub)
    Cr_blocks, _, _ = utils.make_blocks(Cr_sub)

    # 4. DCT & Quantization (成員 C & A)
    t0 = time.time()
    encoded_data = {'Y': [], 'Cb': [], 'Cr': []}
    
    # 為了計算壓縮率，我們需要收集量化後的 block
    quantized_Y_blocks = []
    quantized_C_blocks = []

    # Select functions for pipeline
    forward_dct_fn = _resolve_func(transforms, 'forward_dct_block', dct_method)
    inverse_dct_fn = _resolve_func(transforms, 'inverse_dct_block', idct_method)
    quantize_fn = _resolve_func(entropy, 'quantize', entropy_method)
    dequantize_fn = _resolve_func(entropy, 'dequantize', entropy_method)
    encode_block_fn = _resolve_func(entropy, 'encode_block', entropy_method)
    decode_block_fn = _resolve_func(entropy, 'decode_block', entropy_method)

    # Record chosen methods
    stats['methods'] = {
        'dct': dct_method,
        'idct': idct_method,
        'downsample': downsample_method,
        'upsample': upsample_method,
        'entropy': entropy_method,
    }

    # Process Y
    for block in Y_blocks:
        dct_block = forward_dct_fn(block)
        q_block = quantize_fn(dct_block, config.Q_Y) # 量化
        stream = encode_block_fn(q_block)
        
        encoded_data['Y'].append(stream)
        quantized_Y_blocks.append(q_block) # 收集起來算非零係數

    # Process Cb/Cr
    for block in Cb_blocks:
        dct_block = forward_dct_fn(block)
        q_block = quantize_fn(dct_block, config.Q_C)
        stream = encode_block_fn(q_block)
        encoded_data['Cb'].append(stream)
        quantized_C_blocks.append(q_block)

    for block in Cr_blocks:
        dct_block = forward_dct_fn(block)
        q_block = quantize_fn(dct_block, config.Q_C)
        stream = encode_block_fn(q_block)
        encoded_data['Cr'].append(stream)
        quantized_C_blocks.append(q_block)
        
    stats['time_encoding'] = (time.time() - t0) * 1000

    # --- 計算壓縮指標 ---
    nz_y = utils.count_non_zero(quantized_Y_blocks)
    nz_c = utils.count_non_zero(quantized_C_blocks)
    # 原始大小（未壓縮的 RGB）
    orig_bytes = img_arr.shape[0] * img_arr.shape[1] * 3  # 每像素 3 bytes

    # 估算壓縮大小：優先使用 entropy bitstream 長度
    def _sum_bits(streams):
        total = 0
        for s in streams:
            if isinstance(s, dict) and 'bits' in s and isinstance(s['bits'], str):
                total += len(s['bits'])
            else:
                # 後備估算：以非零係數數量粗略估計，每個非零給固定 12 bits
                # 注意：這只是後備方案，真正壓縮應以 bitstream 為準
                try:
                    total += int(np.count_nonzero(s)) * 12
                except Exception:
                    total += 0
        return total

    total_bits = _sum_bits(encoded_data['Y']) + _sum_bits(encoded_data['Cb']) + _sum_bits(encoded_data['Cr'])
    comp_bytes = total_bits / 8.0
    ratio = (orig_bytes / comp_bytes) if comp_bytes > 0 else float('inf')

    stats['original_bytes'] = orig_bytes
    stats['compressed_bytes'] = comp_bytes
    stats['compression_ratio'] = ratio
    stats['sparsity'] = 100 * (1 - (nz_y + nz_c) / (h*w*1.5)) # 係數為0的比例

    # --- DECODER ---
    t0 = time.time()
    
    # Y Reconstruction
    Y_recon_blocks = []
    for stream in encoded_data['Y']:
        q_block = decode_block_fn(stream)
        dct_block = dequantize_fn(q_block, config.Q_Y)
        spatial_block = inverse_dct_fn(dct_block) # 成員 C (IDCT)
        Y_recon_blocks.append(spatial_block)
    Y_recon = utils.reconstruct_from_blocks(Y_recon_blocks, h_y, w_y)
    
    # Cb/Cr Loop
    Cb_recon_blocks = []
    for stream in encoded_data['Cb']:
        q_block = decode_block_fn(stream)
        dct_block = dequantize_fn(q_block, config.Q_C)
        spatial_block = inverse_dct_fn(dct_block)
        Cb_recon_blocks.append(spatial_block)
    Cb_recon_sub = utils.reconstruct_from_blocks(Cb_recon_blocks, h_c, w_c)

    Cr_recon_blocks = []
    for stream in encoded_data['Cr']:
        q_block = decode_block_fn(stream)
        dct_block = dequantize_fn(q_block, config.Q_C)
        spatial_block = inverse_dct_fn(dct_block)
        Cr_recon_blocks.append(spatial_block)
    Cr_recon_sub = utils.reconstruct_from_blocks(Cr_recon_blocks, h_c, w_c)
    # (Loop 結束)

    stats['time_decoding_dct'] = (time.time() - t0) * 1000

    # 5. Upsampling (成員 B)
    t0 = time.time()
    upsample_fn = _resolve_func(sampler, 'upsample', upsample_method)
    Cb_recon = upsample_fn(Cb_recon_sub, h, w)
    Cr_recon = upsample_fn(Cr_recon_sub, h, w)
    stats['time_upsample'] = (time.time() - t0) * 1000

    # 6. Merge
    recon_img_ycbcr = np.zeros((h, w, 3))
    recon_img_ycbcr[:,:,0] = Y_recon
    recon_img_ycbcr[:,:,1] = Cb_recon
    recon_img_ycbcr[:,:,2] = Cr_recon
    
    result_rgb = utils.ycbcr_to_rgb(recon_img_ycbcr)
    stats['total_time'] = (time.time() - t_start) * 1000

    return result_rgb, stats

def process_image(img_path, output_dir, method_cfg):
    """讀取單張圖片，處理並存檔"""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[Skip] 無法讀取 {img_path.name}: {e}")
        return

    img_arr = np.array(img)
    h, w, _ = img_arr.shape

    # 自動裁切成 16 的倍數 (避免 padding 邏輯複雜化)
    h_new, w_new = (h // 16) * 16, (w // 16) * 16
    if h_new != h or w_new != w:
        print(f"  -> Auto cropping from {w}x{h} to {w_new}x{h_new}")
        img_arr = img_arr[:h_new, :w_new, :]

    # 執行 Pipeline (現在會回傳 stats)
    recon_img, stats = jpeg_pipeline(
        img_arr,
        dct_method=method_cfg.get('dct_method', 'baseline'),
        idct_method=method_cfg.get('idct_method', 'baseline'),
        downsample_method=method_cfg.get('downsample_method', 'baseline'),
        upsample_method=method_cfg.get('upsample_method', 'baseline'),
        entropy_method=method_cfg.get('entropy_method', 'baseline'),
    )

    psnr_val = utils.calculate_psnr(img_arr, recon_img)
    
    # 顯示詳細數據
    print(f"--- {img_path.name} ---")
    print(f"  [Quality] PSNR: {psnr_val:.2f} dB")
    print(f"  [Size]    Original: {stats['original_bytes']/1024:.1f} KB -> Est. Compressed: {stats['compressed_bytes']/1024:.1f} KB")
    print(f"  [Ratio]   Compression Ratio: {stats['compression_ratio']:.2f}x (Sparsity: {stats['sparsity']:.1f}%)")
    print(f"  [Time]    DCT/Quant: {stats['time_encoding']:.1f}ms | IDCT: {stats['time_decoding_dct']:.1f}ms")
    print(f"  [Time]    Upsample: {stats['time_upsample']:.1f}ms | Total: {stats['total_time']:.1f}ms")
    print("-" * 30)
    
    # 存檔 (原圖與結果並排，方便對比)
    compare_img = np.hstack((img_arr, recon_img))
    res_img = Image.fromarray(compare_img.astype(np.uint8))
    
    output_filename = output_dir / f"result_{img_path.stem}.png"
    res_img.save(output_filename)
    
    # 另存 JSON 摘要
    summary = {
        'image': img_path.name,
        'width': img_arr.shape[1],
        'height': img_arr.shape[0],
        'psnr': float(psnr_val),
        'stats': stats,
    }
    json_path = output_dir / f"result_{img_path.stem}.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warn] 無法寫入 JSON: {e}")

    print(f"[Done] {img_path.name} -> PSNR: {psnr_val:.2f} dB")

def main():
    parser = argparse.ArgumentParser(description='JPEG Encoder/Decoder Team Project')
    parser.add_argument('--image_dir', type=str, default='images', help='圖片資料夾')
    parser.add_argument('--image_name', type=str, default=None, help='圖片名稱')
    parser.add_argument('--output_dir', type=str, default='outputs', help='輸出的資料夾路徑')
    # 方法選擇參數（suffix 對應到函式名，如 forward_dct_block_fft 等）
    parser.add_argument('--dct_method', type=str, default='baseline', help='DCT 方法（baseline/自訂後綴）')
    parser.add_argument('--idct_method', type=str, default='baseline', help='IDCT 方法（baseline/自訂後綴）')
    parser.add_argument('--downsample_method', type=str, default='baseline', help='降採樣方法（baseline/bilinear/...）')
    parser.add_argument('--upsample_method', type=str, default='baseline', help='升採樣方法（baseline/bilinear/...）')
    parser.add_argument('--entropy_method', type=str, default='baseline', help='熵編碼方法（baseline/自訂後綴）')
    
    # 這裡如果不使用命令列，也可以直接改這裡的預設值
    # 例如：args = parser.parse_args(['--image_dir', 'my_folder'])
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_dir_name = '/' + args.image_dir.split('/')[-1]
    image_name = args.image_name
    output_dir = Path(args.output_dir + image_dir_name)
    output_dir.mkdir(exist_ok=True, parents=True) # 自動建立輸出資料夾

    print(f"=== JPEG Project Start ===")
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}\n")

    method_cfg = {
        'dct_method': args.dct_method,
        'idct_method': args.idct_method,
        'downsample_method': args.downsample_method,
        'upsample_method': args.upsample_method,
        'entropy_method': args.entropy_method,
    }
    methods_common = {
        'method': method_cfg
    }
    
    print("Common:")
    print(json.dumps(methods_common, ensure_ascii=False, indent=2))

    if image_name:
        # 單張模式
        process_image(image_dir / image_name, output_dir, method_cfg)
    elif image_dir.is_dir():
        # 資料夾模式
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts]
        
        if not images:
            print("資料夾內沒有支援的圖片格式。")
        else:
            print(f"找到 {len(images)} 張圖片，開始批次處理...")
            for img_file in images:
                process_image(img_file, output_dir, method_cfg)
    else:
        print("輸入路徑不存在，請確認路徑是否正確。")

if __name__ == "__main__":
    main()