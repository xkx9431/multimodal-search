"""纯 Python 版 CLIP 编码，不依赖 Spark。"""
import io
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from minio import Minio

meta = pd.read_parquet("output/metadata.parquet")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model = model.to(device).eval()

client = Minio("localhost:9000", access_key="minioadmin",
               secret_key="minioadmin", secure=False)

BATCH = 32
all_vecs = []
keys = meta["cam_front_key"].tolist()

for i in tqdm(range(0, len(keys), BATCH)):
    batch_keys = keys[i:i+BATCH]
    imgs = []
    for k in batch_keys:
        resp = client.get_object("nuscenes-raw", k)
        try:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        finally:
            resp.close()
            resp.release_conn()
        imgs.append(preprocess(img))
    x = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    all_vecs.append(feats.cpu().numpy())

embeddings = np.concatenate(all_vecs, axis=0).astype(np.float32)
print("Embeddings shape:", embeddings.shape)  # (N, 512)

meta["embedding"] = list(embeddings)
meta.to_parquet("output/embeddings.parquet", index=False)
print("Saved -> output/embeddings.parquet")
