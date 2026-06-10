"""把 nuScenes JSON 表转成宽表 Parquet，存到 output/metadata.parquet。"""
from pathlib import Path
import pandas as pd
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)

rows = []
for sample in nusc.sample:
    scene = nusc.get('scene', sample['scene_token'])
    desc = scene['description'].lower()

    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # 是否包含自行车
    has_bicycle = any(
        nusc.get('sample_annotation', tok)['category_name'].startswith('vehicle.bicycle')
        for tok in sample['anns']
    )

    rows.append({
        'sample_token': sample['token'],
        'scene_token': sample['scene_token'],
        'scene_name': scene['name'],
        'scene_description': scene['description'],
        'timestamp': sample['timestamp'],
        'cam_front_key': cam_data['filename'],      # samples/CAM_FRONT/xxx.jpg
        'lidar_top_key': lidar_data['filename'],    # samples/LIDAR_TOP/xxx.pcd.bin
        'is_rainy': 'rain' in desc,
        'is_night': 'night' in desc,
        'has_bicycle': has_bicycle,
    })

df = pd.DataFrame(rows)
print(df.head())
print(f"\nTotal samples: {len(df)}")
print(f"Rainy: {df.is_rainy.sum()}, Night: {df.is_night.sum()}, With bicycle: {df.has_bicycle.sum()}")

Path('output').mkdir(exist_ok=True)
df.to_parquet('output/metadata.parquet', index=False)
print("Saved -> output/metadata.parquet")
