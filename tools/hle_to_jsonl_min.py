#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, json, argparse, hashlib, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tqdm import tqdm

# HF datasets (only needed if --source hf)
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None

import requests


# ----------------- Utils -----------------

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name[:160]

def sha1_16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def save_png(pil_img: Image.Image, out_path: Path) -> None:
    pil_img.convert("RGB").save(out_path, format="PNG")

def fetch_to_file(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


# ----------------- Image extraction -----------------

# Exact keys from your schema + a couple of fallbacks
IMAGE_KEYS_ORDERED = [
    "image_preview",     # PIL.Image (primary)
    "rationale_image",   # PIL.Image (secondary)
    "image",             # string; sometimes URL or unresolved path
    # fallbacks (if your local file has them)
    "image_url", "image_urls", "images"
]

def extract_images(rec: Dict[str, Any]) -> List[Any]:
    imgs: List[Any] = []
    for key in IMAGE_KEYS_ORDERED:
        if key in rec and rec[key] is not None and rec[key] != "":
            v = rec[key]
            if isinstance(v, list):
                imgs.extend(v)
            else:
                imgs.append(v)
    return imgs

def save_images_any(images: List[Any], img_dir: Path, baseid: str) -> List[str]:
    """
    Save images (PIL, HF 'Image' dicts, bytes, local path, or URL) as PNG.
    Returns list of relative filenames (just the basenames).
    """
    saved = []
    for idx, item in enumerate(images):
        try:
            fname = sanitize_filename(f"hle_{baseid}_{idx}.png")
            out_path = img_dir / fname

            # 1) PIL Image directly
            if isinstance(item, Image.Image):
                save_png(item, out_path)

            # 2) HF 'Image' dict-like (rare but possible)
            elif hasattr(item, "keys"):
                if "image" in item and isinstance(item["image"], Image.Image):
                    save_png(item["image"], out_path)
                elif "bytes" in item and item["bytes"] is not None:
                    img = Image.open(io.BytesIO(item["bytes"]))
                    save_png(img, out_path)
                elif "path" in item and item["path"]:
                    with open(item["path"], "rb") as f:
                        b = f.read()
                    img = Image.open(io.BytesIO(b))
                    save_png(img, out_path)
                else:
                    # last-ditch: try to PIL-open
                    img = Image.open(item)  # may raise
                    save_png(img, out_path)

            # 3) String: URL or local path
            elif isinstance(item, str):
                s = item.strip()
                if s.startswith(("http://", "https://")):
                    fetch_to_file(s, out_path)
                elif Path(s).exists():
                    with open(s, "rb") as f:
                        b = f.read()
                    img = Image.open(io.BytesIO(b))
                    save_png(img, out_path)
                else:
                    # Unknown string; skip silently
                    continue

            else:
                # Unknown type; skip
                continue

            saved.append(fname)
        except Exception as e:
            print(f"[warn] failed to save image idx={idx} for {baseid}: {e}", file=sys.stderr)

    return saved


# ----------------- Normalization -----------------

def get_field(rec: Dict[str, Any], key: str, default: str = "") -> str:
    v = rec.get(key, default)
    return v if isinstance(v, str) else default

def build_qid(rec: Dict[str, Any]) -> str:
    if "id" in rec and isinstance(rec["id"], str) and rec["id"]:
        return rec["id"]
    # fallback: hash the question text
    return sha1_16(get_field(rec, "question", ""))


# ----------------- Loaders -----------------

def iter_hf(dataset_id: str, split: str, token: Optional[str]):
    if load_dataset is None:
        raise RuntimeError("datasets library not available. Install with: pip install datasets")
    # token-compatible call across datasets versions
    try:
        ds = load_dataset(dataset_id, split=split, token=token)
    except TypeError:
        ds = load_dataset(dataset_id, split=split, use_auth_token=token)
    for rec in ds:
        yield dict(rec)

def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def iter_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of objects.")
    for rec in data:
        yield rec


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert HLE (choices already in question) to DeepResearch JSONL + images (image_preview/rationale_image)."
    )
    ap.add_argument("--source", choices=["hf", "file"], default="hf",
                    help="Load from HuggingFace (hf) or a local file (file).")
    ap.add_argument("--dataset", default="cais/hle", help="HF dataset id (default: cais/hle)")
    ap.add_argument("--split", default="test", help="HF split (default: test)")
    ap.add_argument("--input", help="Path to local input (.json or .jsonl) when --source file")
    ap.add_argument("--out", default="eval_data/hle.jsonl", help="Output JSONL path")
    ap.add_argument("--img-dir", default="eval_data/file_corpus", help="Directory to save images")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap for quick smoke runs")
    args = ap.parse_args()

    out_path = Path(args.out)
    img_dir = ensure_dir(args.img_dir)
    ensure_dir(out_path.parent)

    # Optional HF token from env (if gated)
    hf_token = os.getenv("HF_TOKEN", None)
    if args.source == "hf" and hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass

    # Loader
    if args.source == "hf":
        it = iter_hf(args.dataset, args.split, hf_token)
    else:
        if not args.input:
            raise ValueError("--input is required when --source file")
        p = Path(args.input)
        if p.suffix.lower() == ".jsonl":
            it = iter_jsonl(p)
        elif p.suffix.lower() == ".json":
            it = iter_json(p)
        else:
            raise ValueError("Input must be .json or .jsonl")

    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(it, desc="converting"):
            if args.limit and written >= args.limit:
                break

            qid = build_qid(rec)
            question_text = get_field(rec, "question", "")
            answer_text   = get_field(rec, "answer", "")

            # pull images with your exact keys
            imgs = extract_images(rec)
            img_files = save_images_any(imgs, img_dir, qid) if imgs else []

            prefix = ((" ".join(img_files) + " ") if img_files else "")
            obj = {
                "question": prefix + question_text,   # prepend filenames if any
                "answer":   answer_text               # unchanged; HLE already provides ground truth
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] wrote {written} lines to {out_path}")
    print(f"[done] images saved under {img_dir.resolve()}")


if __name__ == "__main__":
    main()
