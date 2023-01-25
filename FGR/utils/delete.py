def read_obj_data(LABEL_PATH, calib=None, im_shape=None):
    '''Reads in object label file from Kitti Object Dataset.
        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data
    '''
    used_cls = ['Car', 'Van' ,'Truck', 'Misc']
    objects = []

    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()

    for object_index in range(len(detections)):
        
        data_str = detections[object_index]
        data_list = data_str.split()
        
        if data_list[0] not in used_cls:
            continue

        object_it = KittiObject()

        object_it.cls = data_list[0]
        object_it.truncate = float(data_list[1])
        object_it.occlusion = int(data_list[2])
        object_it.alpha = float(data_list[3])

        #                            width          height         lenth
        object_it.dim = np.array([data_list[9], data_list[8], data_list[10]]).astype(float)

        # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
        object_it.pos = np.array(data_list[11:14]).astype(float) + calib.t_cam2_cam0 # 0.062
        # The orientation definition is inconsitent with right-hand coordinates in kitti
        object_it.orientation = float(data_list[14]) + m.pi/2  
        object_it.R = E2R(object_it.orientation, 0, 0)

        pts3_c_o = []  # 3D location of 3D bounding box corners
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, -object_it.dim[2]])/2.0)

        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0 ], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)

        object_it.boxes[0].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[1].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[2].box = np.array([0.0, 0.0, 0.0, 0.0]).astype(float)
        object_it.boxes[0].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
        object_it.boxes[1].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
        for j in range(2): # left and right boxes
            for i in range(8):
                if pts3_c_o[i][2] < 0:
                    continue
                if j == 0:    # project 3D corner to left image
                    pt2 = Space2Image(calib.p2_2, NormalizeVector(pts3_c_o[i]))
                elif j == 1:  # project 3D corner to right image
                    pt2 = Space2Image(calib.p2_3, NormalizeVector(pts3_c_o[i]))
                if i < 4:
                    object_it.boxes[j].keypoints[i] = pt2[0] 

                object_it.boxes[j].box[0] = min(object_it.boxes[j].box[0], pt2[0])
                object_it.boxes[j].box[1] = min(object_it.boxes[j].box[1], pt2[1]) 
                object_it.boxes[j].box[2] = max(object_it.boxes[j].box[2], pt2[0])
                object_it.boxes[j].box[3] = max(object_it.boxes[j].box[3], pt2[1]) 

            object_it.boxes[j].box[0] = max(object_it.boxes[j].box[0], 0)
            object_it.boxes[j].box[1] = max(object_it.boxes[j].box[1], 0) 

            if im_shape is not None:
                object_it.boxes[j].box[2] = min(object_it.boxes[j].box[2], im_shape[1]-1)
                object_it.boxes[j].box[3] = min(object_it.boxes[j].box[3], im_shape[0]-1)

            # deal with unvisible keypoints
            left_keypoint, right_keypoint = 5000, 0
            left_inx, right_inx = -1, -1
            # 1. Select keypoints that lie on the left and right side of the 2D box
            for i in range(4):
                if object_it.boxes[j].keypoints[i] < left_keypoint:
                    left_keypoint = object_it.boxes[j].keypoints[i]
                    left_inx = i
                if object_it.boxes[j].keypoints[i] > right_keypoint:
                    right_keypoint = object_it.boxes[j].keypoints[i]
                    right_inx = i
            # 2. For keypoints between left and right side, select the visible one
            for i in range(4):
                if i == left_inx or i == right_inx:
                    continue
                if pts3_c_o[i][2] > object_it.pos[2]:
                    object_it.boxes[j].keypoints[i] = -1

        # calculate the union of the left and right box
        object_it.boxes[2].box[0] = min(object_it.boxes[1].box[0], object_it.boxes[0].box[0])
        object_it.boxes[2].box[1] = min(object_it.boxes[1].box[1], object_it.boxes[0].box[1])
        object_it.boxes[2].box[2] = max(object_it.boxes[1].box[2], object_it.boxes[0].box[2])
        object_it.boxes[2].box[3] = max(object_it.boxes[1].box[3], object_it.boxes[0].box[3])

        objects.append(object_it)

    return objects