from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('/home/FGR')
from output.select_most_accurate_pseudolabels import object3d_kitti
from output.select_most_accurate_pseudolabels import select_pseudo_3diou

def get_annos_dict(labels, frame_id, pseudo_labels, idx):
    # This only caters to batch_size of 1
    try: 
        name, truncated, occluded, alpha, bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, dimensions_1, dimensions_2, \
        dimensions_3, location_1, location_2, location_3, rotation_y, score = labels[0].split()
    except: 
        name, truncated, occluded, alpha, bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, dimensions_1, dimensions_2, \
        dimensions_3, location_1, location_2, location_3, rotation_y = labels[0].split()
        score = 1
    name = np.array([name])
    truncated = np.array([float(truncated)], dtype=np.float32)
    occluded = np.array([occluded], dtype=np.float32)
    alpha = np.array([alpha], dtype=np.float32)
    bbox = np.array([[bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4]], dtype=np.float32)
    dimensions = np.array([[dimensions_3, dimensions_1, dimensions_2]], dtype=np.float32)
    location = np.array([[location_1, location_2, location_3]], dtype=np.float32)
    rotation_y = np.array([rotation_y], dtype=np.float32)
    score = np.array([score], dtype=np.float32)

    # For the calculation of boxes_lidar
    pl_obj_list = object3d_kitti.get_objects_from_label(pseudo_labels)
    pl_boxes_lidar = select_pseudo_3diou.labels_to_boxes_lidar([pl_obj_list[idx]], seq)
    
    annos = {}
    annos['name'] = name
    annos['truncated'] = truncated 
    annos['occluded'] = occluded
    annos['alpha'] = alpha 
    annos['bbox'] = bbox
    annos['dimensions'] = dimensions
    annos['location'] = location
    annos['rotation_y'] = rotation_y
    annos['score'] = score         # Ignore the score
    annos['boxes_lidar'] = np.zeros([1,7], dtype=np.float32) #pl_boxes_lidar               # changed/modified
    annos['frame_id'] = frame_id

    return annos

if __name__ == '__main__':

    pl_path =        "/home/hpaat/my_exp/MTrans-evidential/pseudo_label_replace_500_v19.03500.3.fixed_frames/"
    train_ids_path = "/home/hpaat/pcdet/data/kitti/ImageSets/train.txt"

    with open(train_ids_path) as f:
        train_ids = f.readlines()
        train_ids = [x.strip() for x in train_ids]

    det_annos = []
    for seq in tqdm(train_ids, position=0, leave=True):
        
        # Get pseudolabel object list for this sequence
        pseudo_labels = pl_path + seq + '.txt'

        # Read lines
        with open(pseudo_labels) as f:
            label = f.readlines()
            label = [x.strip() for x in label]

        if len(label) == 0:
            continue
        
        for idx, lab in enumerate(label): 
            # Define annos variable
            annos = get_annos_dict([lab], seq, pseudo_labels, idx)

            if len(det_annos) == 0 or seq != det_annos[-1]["frame_id"]:      # If the current frame is not equal to the frame of the last element in det_annos
                det_annos.append(annos)                     
            else: 
                # No need to include additional element to det_annos (just append to the last element)
                det_annos[-1]['name'] = np.concatenate((det_annos[-1]["name"], annos['name']), axis=0)
                det_annos[-1]['truncated'] = np.concatenate((det_annos[-1]["truncated"], annos['truncated']), axis=0)  
                det_annos[-1]['occluded'] =  np.concatenate((det_annos[-1]["occluded"], annos['occluded']), axis=0) 
                det_annos[-1]['alpha'] = np.concatenate((det_annos[-1]["alpha"], annos['alpha']), axis=0)  
                det_annos[-1]['bbox'] = np.concatenate((det_annos[-1]["bbox"], annos['bbox']), axis=0) 
                det_annos[-1]['dimensions'] = np.concatenate((det_annos[-1]["dimensions"], annos['dimensions']), axis=0) 
                det_annos[-1]['location'] = np.concatenate((det_annos[-1]["location"], annos['location']), axis=0) 
                det_annos[-1]['rotation_y'] = np.concatenate((det_annos[-1]["rotation_y"], annos['rotation_y']), axis=0) 
                det_annos[-1]['score'] = np.concatenate((det_annos[-1]["score"], annos['score']), axis=0) 
                det_annos[-1]['boxes_lidar'] = np.concatenate((det_annos[-1]["boxes_lidar"], annos['boxes_lidar']), axis=0)
               
            while det_annos[-1]["frame_id"] != train_ids[len(det_annos)-1]:  
                num_samples = 0
                det_annos.append({
                    'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                    'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                    'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                    'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                    'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
                    'frame_id': train_ids[len(det_annos)-1]
                })
                det_annos[-1], det_annos[-2] = det_annos[-2], det_annos[-1]

    # import pdb; pdb.set_trace() 
    # Save the det_annos for external evaluation in OpenPCDet
    save_file_name = '_train_annos.pkl'
    save_det_annos = True
    if save_det_annos:
        with open('/home/hpaat/my_exp/MTrans-evidential/pseudo_label_replace_500_v19.03500.3.fixed_frames/' + save_file_name, 'wb') as f: 
            pickle.dump(det_annos, f)