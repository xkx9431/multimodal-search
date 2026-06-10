"""上传 nuScenes mini 到 MinIO 的 nuscenes-raw bucket。"""
import os
from pathlib import Path
from minio import Minio
from tqdm import tqdm

MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET = "nuscenes-raw"
LOCAL_ROOT = Path("data/nuscenes")

client = Minio(MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

files = [p for p in LOCAL_ROOT.rglob("*") if p.is_file()]
print(f"Uploading {len(files)} files...")

for fp in tqdm(files):
    key = fp.relative_to(LOCAL_ROOT).as_posix()  # e.g. samples/CAM_FRONT/xxx.jpg
    client.fput_object(BUCKET, key, str(fp))

print("Done.")
