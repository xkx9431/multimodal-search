"""创建 Milvus collection 并导入 CLIP 向量。"""
import pandas as pd
import numpy as np
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema,
    DataType, Collection,
)

COLLECTION = "nuscenes_cam_front"
DIM = 512

connections.connect(alias="default", host="localhost", port="19530")

if utility.has_collection(COLLECTION):
    utility.drop_collection(COLLECTION)

fields = [
    FieldSchema(name="pk",            dtype=DataType.INT64,  is_primary=True, auto_id=True),
    FieldSchema(name="sample_token",  dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="scene_name",    dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="cam_front_key", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="is_rainy",      dtype=DataType.BOOL),
    FieldSchema(name="is_night",      dtype=DataType.BOOL),
    FieldSchema(name="has_bicycle",   dtype=DataType.BOOL),
    FieldSchema(name="embedding",     dtype=DataType.FLOAT_VECTOR, dim=DIM),
]
schema = CollectionSchema(fields, description="nuScenes CAM_FRONT CLIP embeddings")
coll = Collection(name=COLLECTION, schema=schema)

# 索引：HNSW + 内积（向量已归一化 => 余弦相似度）
coll.create_index(
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    },
)

# 插入数据
df = pd.read_parquet("output/embeddings.parquet")
print(f"Inserting {len(df)} rows...")

coll.insert([
    df["sample_token"].tolist(),
    df["scene_name"].tolist(),
    df["cam_front_key"].tolist(),
    df["is_rainy"].tolist(),
    df["is_night"].tolist(),
    df["has_bicycle"].tolist(),
    [np.array(v, dtype=np.float32).tolist() for v in df["embedding"]],
])
coll.flush()
coll.load()

print(f"Inserted. Row count: {coll.num_entities}")
