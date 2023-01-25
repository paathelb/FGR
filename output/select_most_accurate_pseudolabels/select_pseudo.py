import object3d_kitti
import calibration_kitti
import numpy as np
from iou3d_nms_utils import boxes_iou3d_gpu
import torch
from tqdm import tqdm
import pickle
import shutil

def labels_to_boxes_lidar(obj_list, idx):
    calib_file = '/home/hpaat/pcdet/data/kitti/training/calib/' + idx + '.txt'
    #assert calib_file.exists()
    calib = calibration_kitti.Calibration(calib_file)
        
    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)

    loc = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar

    return gt_boxes_lidar


def get_acc_pseudolabel_record():
    train_ids = '/home/hpaat/pcdet/data/kitti/ImageSets/train.txt'
    pl_path = '/home/hpaat/FGR/output/pseudo_labels_orig/labels/'
    gt_path = '/home/hpaat/pcdet/data/kitti/training/label_2/'

    with open(train_ids) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    record_ious = []
    for seq in tqdm(lines, position=0, leave=True):
        pseudo_labels = pl_path + seq + '.txt'
        pl_obj_list = object3d_kitti.get_objects_from_label(pseudo_labels)
        if len(pl_obj_list) == 0:
            continue
        pl_boxes_lidar = labels_to_boxes_lidar(pl_obj_list, seq)
        
        gt_labels = gt_path + seq + '.txt'
        gt_obj_list = object3d_kitti.get_objects_from_label(gt_labels)
        gt_obj_list = [x for x in gt_obj_list if x.cls_type=='Car']
        if len(gt_obj_list) == 0:
            continue
        gt_boxes_lidar = labels_to_boxes_lidar(gt_obj_list, seq)

        iou = boxes_iou3d_gpu(torch.from_numpy(pl_boxes_lidar).float().cuda(), torch.from_numpy(gt_boxes_lidar).float().cuda())

        for idx1, a in enumerate(pl_boxes_lidar):
            for idx2, b in enumerate(gt_boxes_lidar):
                record_ious.append([seq, idx1, iou[idx1, idx2].item()])

    record_ious.sort(key=lambda x:x[2], reverse=True)

    # with open('/home/hpaat/FGR/output/select_most_accurate_pseudolabels/record_ious.pkl', 'wb') as handle:
    #     pickle.dump(record_ious, handle, protocol=pickle.HIGHEST_PROTOCOL)

import os 

def get_acc_pseudolabels():
    with open('/home/hpaat/FGR/output/select_most_accurate_pseudolabels/record_ious.pkl', 'rb') as handle:
        record_ious = pickle.load(handle)

    top_num = 2000
    train_ids = '/home/hpaat/pcdet/data/kitti/ImageSets/train.txt'
    gt_path = '/home/hpaat/KITTI/data_object_label_2/training/label_2/'
    path = "/home/hpaat/FGR/output/select_most_accurate_pseudolabels/"
    name_folder = 'acc_labels'

    if os.path.exists(path+name_folder):
        pass
    else:
        os.mkdir(path + name_folder)

    make_pseudo = True
    if make_pseudo == True:
        seq_num_w_label = set()
        for record in tqdm(record_ious[:top_num], position=0, leave=True):
            seq_num_w_label.add(record[0])
            
            with open(gt_path + record[0] + '.txt') as file:
                lines = file.readlines() 
                lines = [x.strip() for x in lines]
                lines = [x for x in lines if x[:3]=='Car']
                line_to_copy = lines[record[1]]
            
            if os.path.exists(path + name_folder + '/' + record[0] + '.txt'):
                os.remove(path + name_folder + '/' + record[0] + '.txt')
            with open(path + name_folder + '/' + record[0] + '.txt', 'a') as file:
                file.write(line_to_copy)
    
    import pdb; pdb.set_trace() 
    seq_num_w_label = list(seq_num_w_label)
    with open(train_ids) as file:
        train_ids_ = file.readlines()
        train_ids_ = [x.strip() for x in train_ids_]

    for idx in tqdm(train_ids_, position=0, leave=True):
        if idx not in seq_num_w_label:
            with open(path + name_folder + '/' + idx + '.txt', 'a') as file:
                file.write('')

def get_val_gt_labels():
    val_ids = '/home/hpaat/pcdet/data/kitti/ImageSets/val.txt'
    gt_path = '/home/hpaat/KITTI/data_object_label_2/training/label_2/'
    path = "/home/hpaat/FGR/output/select_most_accurate_pseudolabels/"
    name_folder = 'acc_labels'

    with open(val_ids) as file:
        val_ids_ = file.readlines()
        val_ids_ = [x.strip() for x in val_ids_] 

    for idx in tqdm(val_ids_, position=0, leave=True):
        source = gt_path + idx + '.txt'
        dest = path + name_folder + '/' + idx + '.txt'
        shutil.copy(source, dest)






  

    
