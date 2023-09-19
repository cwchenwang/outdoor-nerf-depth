from extract_sfm import read_model, parse_camera_dict
from normalize_cam_dict import *
from tqdm import tqdm
from PIL import Image

kitti_dict = {
    'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s657_e787_densegt': {
        'start': 657,
        'end': 787,
        'drive': '2011_10_03_drive_0027_sync',
    },
    'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s890_e1028_densegt': {
        'start': 890,
        'end': 1028,
        'drive': '2011_10_03_drive_0027_sync',
    },
    'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s1052_e1392_densegt': {
        'start': 1052,
        'end': 1392,
        'drive': '2011_10_03_drive_0027_sync',
    },
    'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt': {
        'start': 2700,
        'end': 3000,
        'drive': '2011_10_03_drive_0027_sync',
    },
    'KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s1307_e1672_densegt': {
        'start': 1307,
        'end': 1672,
        'drive': '2011_10_03_drive_0034_sync',
    },
    'KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2418_e2748_densegt': {
        'start': 2418,
        'end': 2748,
        'drive': '2011_10_03_drive_0034_sync',
    },
    'KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt': {
        'start': 2749,
        'end': 2929,
        'drive': '2011_10_03_drive_0034_sync',
    },
    'KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt': {
        'start': 400,
        'end': 725,
        'drive': '2011_09_30_drive_0018_sync',
    },
    'KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s1500_e1858_densegt': {
        'start': 1500,
        'end': 1858,
        'drive': '2011_09_30_drive_0018_sync',
    },
    'KITTISeq06_2011_09_30_drive_0020_sync_llffdtu_s330_e805_densegt': {
        'start': 330,
        'end': 805,
        'drive': '2011_09_30_drive_0020_sync',
    }
}

def extract_all_to_dir(sparse_dir, out_dir, seq_len, ext='.bin'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    camera_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    xyz_file = os.path.join(out_dir, 'kai_points.txt')
    track_file = os.path.join(out_dir, 'kai_tracks.json')
    keypoints_file = os.path.join(out_dir, 'kai_keypoints.json')
    
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)

    camera_dict_all = parse_camera_dict(colmap_cameras, colmap_images)
    # import pdb; pdb.set_trace();
    camera_dict = {}
    # for i in range(2700-5, 3000-5):
    for i in range(0, seq_len):
        camera_dict[f'{i:08d}.png'] = camera_dict_all[f'{i:08d}.png']
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

offset = 5
for seq in kitti_dict.keys():
  in_dir = '/ssd1/wangchen/data/10.109.83.30:8021/users/wangchen/kitti_select_static_10seq'
  sparse_dir = os.path.join(in_dir, seq, 'sparse/0')
  out_dir = os.path.join('/ssd1/wangchen/data/nerfpp_preprocessed', seq)
  out_cam_dict_file = os.path.join(out_dir, 'kai_cameras_norm.json')
  if not os.path.exists(os.path.join(out_dir, out_cam_dict_file)):
    extract_all_to_dir(sparse_dir, out_dir, kitti_dict[seq]['end'] - kitti_dict[seq]['start'] - offset, ext='.txt')
    in_cam_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    scale = normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1)
    with open(os.path.join(base_dir, 'scale'), 'w') as file:
      file.write(str(scale) + '\n')
    print('produce camera dict finished')
  with open(os.path.join(out_cam_dict_file)) as fp:
    camera_dict = json.load(fp)
  base_dir = out_dir
  train_dir = os.path.join(base_dir, 'train')
  test_dir = os.path.join(base_dir, 'test')
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)
  for d in ['intrinsics', 'pose', 'rgb']:
    for split in ['train', 'test']:
      os.makedirs(os.path.join(base_dir, split, d), exist_ok=True)
  depth_types = ['mono_crop', 'ste_conf_-1_crop', 'ste_conf_0.95_crop']
  suffix_func = lambda sfx: '_'+sfx if sfx!='gt' else ''
  for depth_type in depth_types:
    for split in ['train', 'test']:
      os.makedirs(os.path.join(base_dir, split, f'depth{suffix_func(depth_type)}'), exist_ok=True)
  k, l = 0, 0

  start, end = kitti_dict[seq]['start'], kitti_dict[seq]['end']
  seq_len = end - start - offset
  test_indices = [i for i in range(9, seq_len, 10)]
  for i in tqdm(range(seq_len)):
    data = camera_dict[f'{i:08}.png']
    folder = test_dir if i in test_indices else train_dir
    if i in test_indices:
        idx = k
        k += 1
    else:
        idx = l
        l += 1
    # with open(os.path.join(folder, 'intrinsics', f'{idx:05d}.txt'), 'w') as file:
    #     file.write(' '.join(str(k) for k in data["K"]))
    # with open(os.path.join(folder, 'pose', f'{idx:05d}.txt'), 'w') as file:
    #     file.write(' '.join(str(k) for k in data["C2W"])) 
    # Image.open(f'{in_dir}/{seq}/images/{i:08}.png').save(os.path.join(folder, 'rgb', f'{idx:05d}.png'))  
    for depth_type in ['mono_crop', 'ste_conf_-1_crop', 'ste_conf_0.95_crop']:
    #   import pdb; pdb.set_trace();
      Image.open(f'{in_dir}/{seq}/depths_{depth_type}/{i:08}.png').save(os.path.join(folder, f'depth{suffix_func(depth_type)}', f'{idx:05d}.png'))