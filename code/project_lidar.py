"""把 LIDAR_TOP 的点云投影到 CAM_FRONT 图像上，按深度上色，保存到 output/projections/。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
out_dir = Path('output/projections')
out_dir.mkdir(parents=True, exist_ok=True)

# 取前 10 个 sample 做演示
for sample in nusc.sample[:10]:
    cam_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']

    # 用 devkit 一行搞定投影（内部已处理 ego pose + sensor calibration）
    points, coloring, im = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=cam_token,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=2, cmap='jet')
    ax.axis('off')

    out_path = out_dir / f"{sample['token']}.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")
