"""Spark + Pandas UDF 批量提取 CLIP 图像特征，结果写到 MinIO 的 Parquet。"""
import io
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType
from minio import Minio

# ---- Spark 连接 MinIO 当 S3A ----
spark = (
    SparkSession.builder
    .appName("clip-encode")
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

# ---- 读取元数据宽表 ----
meta = spark.read.parquet("output/metadata.parquet").select("sample_token", "cam_front_key")
print(f"To encode: {meta.count()} images")

# ---- Pandas UDF：每个 partition 加载一次模型 ----
@pandas_udf(ArrayType(FloatType()))
def clip_encode_udf(keys: pd.Series) -> pd.Series:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device).eval()

    client = Minio("localhost:9000", access_key="minioadmin",
                   secret_key="minioadmin", secure=False)

    vectors = []
    batch_imgs = []
    BATCH = 32

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        for v in feats.cpu().numpy():
            vectors.append(v.astype(np.float32).tolist())
        batch_imgs.clear()

    for key in keys:
        resp = client.get_object("nuscenes-raw", key)
        try:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        finally:
            resp.close()
            resp.release_conn()
        batch_imgs.append(preprocess(img))
        if len(batch_imgs) >= BATCH:
            flush()
    flush()
    return pd.Series(vectors)

encoded = meta.withColumn("embedding", clip_encode_udf(col("cam_front_key")))
encoded.write.mode("overwrite").parquet("output/embeddings.parquet")
print("Saved -> output/embeddings.parquet")
spark.stop()
