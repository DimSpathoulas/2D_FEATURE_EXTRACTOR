from tqdm import tqdm
from nuscenes import NuScenes
import numpy as np
import cv2
import copy
from pyquaternion import Quaternion
from mrcnn.config import Config
from mrcnn import my_model_v3 as modellib
from mrcnn import visualize
import os
import torch
from PIL import Image
import argparse
import os.path as osp
import pickle
from shapely.geometry import MultiPoint, box
from shapely.geometry import Polygon, LineString

# APO TO NUSCENES
tracking_names = ['pedestrian', 'bicycle', 'motorcycle', 'car', 'bus', 'truck', 'trailer']

detector_classes = ['bg', 'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def rot_z(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def create_box(bbox3d_input):
    # [x, y, z, w, l, h, rot]

    bbox3d = copy.copy(bbox3d_input)

    R = rot_z(bbox3d[6])

    w = bbox3d[3]
    l = bbox3d[4]
    h = bbox3d[5]

    # print('w=', w, 'l=', l)

    # 3d bounding box corners
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    # z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
    z_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)


def retrieve_box_info(transformed_corners):
    # Transpose the array if needed

    transformed_corners = transformed_corners.T

    # Compute the center of the box
    center_x = np.mean(transformed_corners[:, 0])
    center_y = np.mean(transformed_corners[:, 1])
    center_z = np.mean(transformed_corners[:, 2])

    dx = transformed_corners[0, 0] - transformed_corners[1, 0]
    dy = transformed_corners[0, 1] - transformed_corners[1, 1]

    rot_z = np.arctan2(dy, dx)
    
    # dxx = transformed_corners[1, 0] - transformed_corners[0, 0]
    # dyy = transformed_corners[1, 1] - transformed_corners[0, 1]
    # rot_z2 = np.arctan2(dyy, dxx)
    # print(rot_z2 - rot_z)

    x = np.array([center_x, center_y, center_z, rot_z])
    return x


def initialize_model(model_dir=''):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)

    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


def detect_objects(model, image):
    r = model.detect([image], verbose=0)[0]  # h 1 den jero akoma
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return r


class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "inference"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 80
    # Set the GPU to use, if available
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def post_process_coords(corner_coords, imsize=[1600, 900]):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)

        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        elif isinstance(img_intersection, LineString):
            intersection_coords = np.array([coord for coord in img_intersection.coords])
        else:
            # Handle other intersection types if needed
            intersection_coords = None

        if intersection_coords is not None:
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
            return min_x, min_y, max_x, max_y

    return None


def lidar_to_world(nusc, lidar_data, box):

    # 1 from lidar to ego
    cs_record = nusc.get(
        'calibrated_sensor', lidar_data['calibrated_sensor_token'])
    box[:3, :] = np.dot(Quaternion(
        cs_record['rotation']).rotation_matrix, box[:3, :])
    for b in range(3):
        box[b, :] = box[b, :] + cs_record['translation'][b]

    # 2 from ego to world based on lidar
    poserecord = nusc.get(
        'ego_pose', lidar_data['ego_pose_token'])
    box[:3, :] = np.dot(Quaternion(
        poserecord['rotation']).rotation_matrix, box[:3, :])
    for b in range(3):
        box[b, :] = box[b, :] + poserecord['translation'][b]

    return box


def world_to_cam(nusc, cam, box):

    # 3 from world to ego based on cam
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    for b in range(3):
        box[b, :] = box[b, :] - poserecord['translation'][b]
    box[:3, :] = np.dot(Quaternion(
        poserecord['rotation']).rotation_matrix.T, box[:3, :])

    # 4 from ego to camera
    cs_record = nusc.get(
        'calibrated_sensor', cam['calibrated_sensor_token'])
    for b in range(3):
        box[b, :] = box[b, :] - cs_record['translation'][b]
    box[:3, :] = np.dot(Quaternion(
        cs_record['rotation']).rotation_matrix.T, box[:3, :])

    # 5 project to image
    points = box
    depths = points[2, :]

    viewpad = np.eye(4)
    viewpad[:np.array(
        cs_record['camera_intrinsic']).shape[0], :np.array(
        cs_record['camera_intrinsic']).shape[1]] = np.array(
        cs_record['camera_intrinsic'])

    nbr_points = points.shape[1]
    # Do operation in homogenous coordinates.
    points = np.concatenate(
        (points[:3, :], np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    points = points / points[2:3,
                      :].repeat(3, 0).reshape(3, nbr_points)

    return points, depths


def save_results(new_data, base_filename):
    # Check if the file exists
    if os.path.exists(base_filename):
        # Load existing data
        with open(base_filename, 'rb') as read_file:
            existing_data = pickle.load(read_file)

        # Ensure existing_data is a dictionary
        if not isinstance(existing_data, dict):
            existing_data = {}
    else:
        existing_data = {}

    # Merge the new data with existing data
    existing_data.update(new_data)

    # Save the combined data back to the file
    with open(base_filename, 'wb') as write_file:
        pickle.dump(existing_data, write_file)

def main():

    parser = argparse.ArgumentParser(description="Project 3d detections to camera planes and extract feature vectors.")

    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version --v1.0-trainval or v1.0-mini')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--detection_file', type=str,
                        default="/home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_val_057.npy",
                        help='Path to the npy detection file')
    parser.add_argument('--output_file', type=str,
                        default='mrcnn_val_057_nothing.pkl',
                        help='Path to the output pkl file')

    args = parser.parse_args()

    data = np.load(args.detection_file, allow_pickle=True)
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    model = initialize_model(model_dir='lgs')

    output_file_pkl = args.output_file
    results = {}

    # for all scenes
    for i in tqdm(range(len(data))):
          
        # retrieve the sample_token and timestamp
        sample_token = data[i]['metadata'][0]['token']
        # print(sample_token)
        results[sample_token] = []

        timestamp = data[i]['metadata'][0]['timestamp']
        num_objects = data[i]['pred_labels'].shape[0]

        # retrieve the lidar to ego transformation
        sample_data = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample_data['data']['LIDAR_TOP'])

        camera_vector = {}

        # for each camera ONLY ONCE not really needed anymore
        for v, sensor in enumerate(sensors):
            cam_data = nusc.get('sample_data', sample_data['data'][sensor])

            im = Image.open(osp.join(nusc.dataroot, cam_data['filename']))
            opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            camera_vector[v] = {
                'image': opencvImage,
                'mrcnn_results': None,
                'im': im,
                'cam': cam_data
            }

        # Project
        projection_dict = {}
        dict_info_stack = {}
        camera_onehot_vec = [0] * len(sensors)

        # for every camera ( or v )
        for v, cam_info in camera_vector.items():

            projection_dict[v] = np.empty((0, 4))
            dict_info_stack[v] = np.empty((0, 1))
            camera_onehot_vec[v] = 1

            image = cam_info['image']
            cam = cam_info['cam']
            im = cam_info['im']

            # for each object
            for j in range(num_objects):

                label = data[i]['pred_labels'][j].cpu().numpy()
                track_name = detector_classes[label]
                if track_name not in tracking_names:
                    continue

                # retrieve information
                pred_box = data[i]['pred_boxes'][j].cpu().numpy()

                # create box
                box = create_box(pred_box).T

                box = lidar_to_world(nusc, lidar_data, box)

                pred_box_worlds = retrieve_box_info(transformed_corners=box)

                final_box_preds = np.array((pred_box_worlds[0], pred_box_worlds[1], pred_box_worlds[2],
                                          pred_box[3], pred_box[4], pred_box[5], pred_box_worlds[3],
                                           pred_box[7], pred_box[8]))

                # print('final[3], [4]', final_box_preds[3], final_box_preds[4])
                # print(final_box_pred, 'daa', final_box_pred)

                we_cooked = create_box(final_box_preds).T

                points, depths = world_to_cam(nusc, cam, we_cooked)

                # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
                # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
                # casing for non-keyframes which are slightly out of sync.
                mask = np.ones(depths.shape[0], dtype=bool)
                mask = np.logical_and(mask, depths > 1.0)
                # mask = np.logical_and(mask, points[0, :] > 1)
                # mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
                # mask = np.logical_and(mask, points[1, :] > 1)
                # mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
                points = points[:, mask]

                points = points[:2, :].T

                # mporei na xalaei ligo to feature vector consistency alla einai ok mallon genika
                # H META MPOROYME NA PAROYME EAN EINAI TO IDIO KENTRO TOY 3D TO MEGALYTERO AREA APO AYTO KRATAME TO FEAT.V
                if points.shape[0] <= 2:
                    continue

                final_coords = post_process_coords(points)

                if final_coords is None:
                    continue

                # y1, x1, y2, x2 = rois[id]
                min_x, min_y, max_x, max_y = [int(coord) for coord in final_coords]

                min_x = max(min_x, 4)
                min_y = max(min_y, 4)
                max_x = min(max_x, 1596)
                max_y = min(max_y, 896)

                projection = np.array([[min_y, min_x, max_y, max_x]])

                projection_dict[v] = np.vstack((projection_dict[v], projection))
                dict_info_stack[v] = np.vstack((dict_info_stack[v], j))

                # ## Draw rectangle on image with coords
                # image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 165, 0), 3)
                #
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 1
                # font_thickness = 1
                # text_size = cv2.getTextSize(track_name, font, font_scale, font_thickness)[0]
                # text_x = int((min_x + max_x - text_size[0]) / 2)  # Center text horizontally
                # text_y = min_y - 10  # Place text above the rectangle, adjust if needed
                # cv2.putText(image, track_name, (text_x, text_y), font, font_scale,
                #             (255, 255, 255), font_thickness,
                #             cv2.LINE_AA)
                #
                # for p in range(points.shape[0]):
                #     cv2.circle(
                #         image, (int(points[p][0]), int(points[p][1])),
                #         6, (0, 255, 0), -1)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONT DELETE                  cam_info['image'] = image.copy()

                cam_info['image'] = image.copy()

        results_temp = {trckname: [] for trckname in tracking_names}

        # save
        for v, cam_info in camera_vector.items():

            # if not (v == 0 or v == 5):
            #     continue

            camera_onehot_vec = [0] * len(sensors)
            camera_onehot_vec[v] = 1

            image = cam_info['image']
            projections = projection_dict[v]
            jj = dict_info_stack[v]

            mrcnn_results = model.detect([image], projections=projections, verbose=0)
            feature_vectors = mrcnn_results

            # print(camera_onehot_vec, feature_vectors.shape[0])
            for step, jjj in enumerate(jj):

                cap = 0

                Projection = projections[step]
                label = data[i]['pred_labels'][jjj].cpu().numpy()
                label = label[0]
                track_name = detector_classes[label]

                point_cloud_feats = data[i]['features'][jjj].cpu().numpy().reshape(512, 3, 3)

                pred_score = data[i]['pred_scores'][jjj].cpu().numpy()

                pred_boxes = data[i]['pred_boxes'][jjj].cpu().numpy().reshape(9)
                lidar_world = create_box(pred_boxes).T
                world_box = lidar_to_world(nusc, lidar_data, lidar_world)  # MPOREI TO ROT NA NE LIGO PERIERGO EDO
                pred_box_world = retrieve_box_info(world_box)

                # x, y, z, w, l, h, rot_z, dx, dy in world coords
                final_box_pred = np.array((pred_box_world[0], pred_box_world[1], pred_box_world[2],
                                           pred_boxes[3], pred_boxes[4], pred_boxes[5], pred_box_world[3],
                                           pred_boxes[7], pred_boxes[8]))

                # print('finalklll[3], [4]', final_box_pred[3], final_box_pred[4])

                for existing_result in results_temp[track_name]:

                    if np.all(existing_result['box'] == final_box_pred):
                        cap = 1
                        # print(track_name, existing_result['box'], pred_boxes, camera_onehot_vec)
                        old_projection = existing_result['projection']

                        old_area = ((old_projection[2] - old_projection[0]) *
                                    (old_projection[3] - old_projection[1]))
                        new_area = ((Projection[2] - Projection[0]) *
                                    (Projection[3] - Projection[1]))

                        if new_area > old_area:
                            existing_result['projection'] = Projection
                            existing_result['point_cloud_features'] = point_cloud_feats
                            existing_result['feature_vector'] = feature_vectors[step]
                            existing_result['camera_onehot_vector'] = camera_onehot_vec
                            existing_result['pred_score'] = pred_score

                if cap == 0:

                    results_temp[track_name].append({
                        'sample_token': sample_token,  # for error checks
                        'timestamp': timestamp,  # for error checks
                        'track_name': track_name,  # for error checks
                        'box': final_box_pred,
                        'projection': Projection,
                        'point_cloud_features': point_cloud_feats,
                        'feature_vector': feature_vectors[step],
                        'camera_onehot_vector': camera_onehot_vec,
                        'pred_score': pred_score
                    })

                print(results_temp, '\n\n\n\n\n\n\n')
            # ### for vis
            # cv2.imshow(str(v), cam_info['image'])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        results[sample_token].append(results_temp)
        if i % 500 == 0:
            save_results(results, output_file_pkl)
            results = {}
          
    if results:
        save_results(results, output_file_pkl)


if __name__ == "__main__":
    main()
