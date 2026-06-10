"""把检索包装成 HTTP 服务。"""
from datetime import timedelta
import io
from typing import Optional
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import open_clip
from pymilvus import connections, Collection
from minio import Minio

COLLECTION = "nuscenes_cam_front"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()

connections.connect(host="localhost", port="19530")
coll = Collection(COLLECTION)
coll.load()
minio_client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

app = FastAPI(title="Multimodal Search")

def _encode_text(text: str) -> np.ndarray:
    with torch.no_grad():
        v = model.encode_text(tokenizer([text]).to(device))
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)[0]

def _encode_image(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    with torch.no_grad():
        v = model.encode_image(preprocess(img).unsqueeze(0).to(device))
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)[0]

def _search(vec: np.ndarray, top_k: int, expr: Optional[str]) -> list:
    res = coll.search(
        data=[vec.tolist()], anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k, expr=expr,
        output_fields=["sample_token", "scene_name", "cam_front_key",
                       "is_night", "has_bicycle"],
    )
    hits = []
    for hit in res[0]:
        key = hit.entity.get("cam_front_key")
        url = minio_client.presigned_get_object(
            "nuscenes-raw", key, expires=timedelta(hours=1)
        )
        hits.append({
            "score": float(hit.score),
            "scene": hit.entity.get("scene_name"),
            "night": hit.entity.get("is_night"),
            "bicycle": hit.entity.get("has_bicycle"),
            "key": key,
            "image_url": url,
        })
    return hits

@app.post("/search/text")
def search_text(
    text: str = Form(...),
    top_k: int = Form(5),
    expr: Optional[str] = Form(None),
):
    return {"results": _search(_encode_text(text), top_k, expr)}

@app.post("/search/image")
def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    expr: Optional[str] = Form(None),
):
    return {"results": _search(_encode_image(file.file.read()), top_k, expr)}
