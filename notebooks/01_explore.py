"""快速验证 nuScenes devkit 能正常读取 mini 数据集。"""
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
print(f"Scenes: {len(nusc.scene)}, Samples: {len(nusc.sample)}")
nusc.list_scenes()
