from extract_sfm import read_model, parse_camera_dict
from normalize_cam_dict import *
from tqdm import tqdm
from PIL import Image

kitti_dict = {
    '2c07fcda-6671-3ac0-ac23-4a232e0e031e': {
        'start': 657,
        'end': 787,
        'drive': '2011_10_03_drive_0027_sync',
        'len': 73
    },
    '70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c': {
        'start': 890,
        'end': 1028,
        'drive': '2011_10_03_drive_0027_sync',
        'len': 72
    },
    'bae67a44-0f30-30c1-8999-06fc1c7ab80a': {
        'start': 1052,
        'end': 1392,
        'drive': '2011_10_03_drive_0027_sync',
        'len': 145
    },
    'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8': {
        'start': 2700,
        'end': 3000,
        'drive': '2011_10_03_drive_0027_sync',
        'len': 73
    },
    'ff78e1a3-6deb-34a4-9a1f-b85e34980f06': {
        'start': 1307,
        'end': 1672,
        'drive': '2011_10_03_drive_0034_sync',
        'len': 73
    },
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
        camera_dict[f'{i:08d}.jpg'] = camera_dict_all[f'{i:08d}.jpg']
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

offset = 0
for seq in kitti_dict.keys():
  in_dir = '/root/paddlejob/workspace/wangchen/data/argo'
  sparse_dir = os.path.join(in_dir, seq, 'sparse/0')
  out_dir = os.path.join('/root/paddlejob/workspace/wangchen/data/argo_nerfpp', seq)
  out_cam_dict_file = os.path.join(out_dir, 'kai_cameras_norm.json')
  if not os.path.exists(os.path.join(out_dir, out_cam_dict_file)):
    extract_all_to_dir(sparse_dir, out_dir, kitti_dict[seq]['len'] - offset, ext='.txt')
    in_cam_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    scale = normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1)
    with open(os.path.join(out_dir, 'scale'), 'w') as file:
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
#   depth_types = ['mono_crop', 'ste_conf_-1_crop', 'ste_conf_0.95_crop']
  depth_types = ['gt']
  suffix_func = lambda sfx: '_'+sfx if sfx!='gt' else ''
  for depth_type in depth_types:
    for split in ['train', 'test']:
      os.makedirs(os.path.join(base_dir, split, f'depth{suffix_func(depth_type)}'), exist_ok=True)
  k, l = 0, 0

  start, end = kitti_dict[seq]['start'], kitti_dict[seq]['end']
  seq_len = kitti_dict[seq]['len']
  test_indices = [i for i in range(9, seq_len, 10)]
  for i in tqdm(range(seq_len)):
    data = camera_dict[f'{i:08}.jpg']
    folder = test_dir if i in test_indices else train_dir
    if i in test_indices:
        idx = k
        k += 1
    else:
        idx = l
        l += 1
    with open(os.path.join(folder, 'intrinsics', f'{idx:05d}.txt'), 'w') as file:
        file.write(' '.join(str(k) for k in data["K"]))
    with open(os.path.join(folder, 'pose', f'{idx:05d}.txt'), 'w') as file:
        file.write(' '.join(str(k) for k in data["C2W"])) 
    Image.open(f'{in_dir}/{seq}/images/{i:08}.jpg').save(os.path.join(folder, 'rgb', f'{idx:05d}.png'))  
    for depth_type in ['gt']:
    #   import pdb; pdb.set_trace();
      Image.open(f'{in_dir}/{seq}/depths_{depth_type}/{i:08}.png').save(os.path.join(folder, f'depth{suffix_func(depth_type)}', f'{idx:05d}.png'))