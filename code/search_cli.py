"""命令行跨模态检索 demo。"""
import argparse
import io
from typing import Optional
import numpy as np
import torch
from PIL import Image
import open_clip
from pymilvus import connections, Collection
from minio import Minio

COLLECTION = "nuscenes_cam_front"

def get_text_vec(text: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tok = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device).eval()
    with torch.no_grad():
        v = model.encode_text(tok([text]).to(device))
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)[0]

def search(vec: np.ndarray, top_k: int = 5, expr: Optional[str] = None) -> None:
    connections.connect(host="localhost", port="19530")
    coll = Collection(COLLECTION)
    coll.load()
    results = coll.search(
        data=[vec.tolist()],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        expr=expr,
        output_fields=["sample_token", "scene_name", "cam_front_key",
                       "is_night", "has_bicycle"],
    )
    for hit in results[0]:
        print(f"score={hit.score:.4f}  scene={hit.entity.get('scene_name')}  "
              f"night={hit.entity.get('is_night')}  bicycle={hit.entity.get('has_bicycle')}  "
              f"key={hit.entity.get('cam_front_key')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="文本查询")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--filter", type=str, default=None,
                        help="Milvus 布尔表达式，例如 is_night == true")
    args = parser.parse_args()

    vec = get_text_vec(args.text)
    search(vec, top_k=args.top, expr=args.filter)
