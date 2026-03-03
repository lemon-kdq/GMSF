#!/usr/bin/env python3
import argparse
from pathlib import Path

# 支持的图片后缀
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def main():
    parser = argparse.ArgumentParser(description="Extract filenames (without suffix) from pcd or image files")
    parser.add_argument("input_dir", help="input folder with pcd or image files")
    parser.add_argument("output_dir", help="output folder to save timestamps txt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    assert input_dir.is_dir(), f"Input directory does not exist: {input_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.is_file()]
    if not files:
        print(f"No files found in {input_dir}")
        return

    # 判断类型
    ext = files[0].suffix.lower()
    if ext == ".pcd":
        out_file = output_dir / "lid_timestamps.txt"
    elif ext in IMG_EXTS:
        out_file = output_dir / "img_timestamps.txt"
    else:
        print(f"Unsupported file type: {ext}")
        return

    # 提取文件名（不含后缀），完全保留文件名里“.pcd/.png”前的所有字符
    names = [f.stem for f in files]
    names.sort()  # 按字典序排序

    # 保存到 txt
    with out_file.open("w") as fo:
        for name in names:
            fo.write(f"{name}\n")

    print(f"File names saved to {out_file} (count={len(names)})")

if __name__ == "__main__":
    main()