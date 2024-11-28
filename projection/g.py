from tqdm import tqdm
from nuscenes import NuScenes
import numpy as np
import cv2
import copy
import multiprocessing as mp
from functools import partial
import os.path as osp
from pyquaternion import Quaternion
from mrcnn.config import Config
from mrcnn import model_mrcnn as modellib
from mrcnn import visualize
import os
import torch
from PIL import Image
import argparse
import pickle
from multiprocessing import Queue
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


def process_camera(v, camera_vector, nusc, lidar_data, scene_data, num_objects, model, sample_token, timestamp):
    projection_dict = {v: np.empty((0, 4))}
    dict_info_stack = {v: np.empty((0, 1))}
    camera_onehot_vec = [0] * len(sensors)
    camera_onehot_vec[v] = 1

    cam_info = camera_vector[v]
    image = cam_info['image']
    cam = cam_info['cam']
    im = cam_info['im']

    results_temp = {trckname: [] for trckname in tracking_names}

    for j in range(num_objects):

        label = scene_data['pred_labels'][j].cpu().numpy()
        track_name = detector_classes[label]

        if track_name not in tracking_names:
            continue

        # retrieve information
        pred_box = scene_data['pred_boxes'][j].cpu().numpy()

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

        
        mask = np.logical_and(mask, depths > 1.0)  # CHANGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

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

    # Process MRCNN results
    projections = projection_dict[v]
    jj = dict_info_stack[v]
    mrcnn_results = model.detect([image], projections=projections, verbose=0)
    feature_vectors = mrcnn_results

    for step, jjj in enumerate(jj):

        cap = 0

        Projection = projections[step]
        label = scene_data['pred_labels'][jjj].cpu().numpy()
        label = label[0]
        track_name = detector_classes[label]

        point_cloud_feats = scene_data['features'][jjj].cpu().numpy().reshape(512, 3, 3)

        pred_score = scene_data['pred_scores'][jjj].cpu().numpy()

        pred_boxes = scene_data['pred_boxes'][jjj].cpu().numpy().reshape(9)
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
                # print('here')
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

    return results_temp


def process_scene(scene_data, nusc, model):

    sample_token = scene_data['metadata'][0]['token']
    timestamp = scene_data['metadata'][0]['timestamp']
    num_objects = scene_data['pred_labels'].shape[0]

    sample_data = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample_data['data']['LIDAR_TOP'])

    camera_vector = {}

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

    results_temp = {trckname: [] for trckname in tracking_names}

    for v in camera_vector.keys():
        cam_results = process_camera(v, camera_vector, nusc, lidar_data, 
                        scene_data, num_objects, model,
                                    sample_token, timestamp)
        
        for trckname, objects in cam_results.items():
            results_temp[trckname].extend(objects)

    return sample_token, results_temp


def load_entire_dataset(file_path):
    print("Loading dataset...")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded dataset with {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def process_scene_queue(scene_data, nusc, model, result_queue):
    result = process_scene(scene_data, nusc, model)
    result_queue.put(result)


def main():

    parser = argparse.ArgumentParser(description="Project 3d detections to camera planes and extract feature vectors.")

    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version --v1.0-trainval or v1.0-mini')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--detection_file', type=str,
                        default="/home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_val_2.npy",
                        help='Path to the npy detection file')
    parser.add_argument('--output_file', type=str,
                        default='mrcnn_val_2_depth2.pkl',
                        help='Path to the output pkl file')
    parser.add_argument('--model_dir', type=str, default='lgs',
                        help='Directory containing the model')
    parser.add_argument('--max_processes', type=int, default=mp.cpu_count() - 2,
                        help='Maximum number of concurrent processes')
    parser.add_argument('--chunk_size', type=int, default=600,
                        help='Number of scenes to process in each chunk')
    # parser.add_argument('--split', type=str, choices=['train', 'val'], required=True,
    #                     help='Specify whether this is the train or val split')
    
    args = parser.parse_args()

    entire_dataset = load_entire_dataset(args.detection_file)
    if entire_dataset is None:
        print("Failed to load the dataset. Exiting.")
        return
        
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    model = initialize_model(model_dir=args.model_dir)

    total_scenes = len(entire_dataset)
    output_file_pkl = args.output_file

    mp.set_start_method('spawn', force=True)
    print('Starting')
   # Use a context manager for the pool
    with mp.Pool(processes=args.max_processes) as pool:
        for chunk_start in tqdm(range(0, total_scenes, args.chunk_size)):
            chunk_end = min(chunk_start + args.chunk_size, total_scenes)
            data_chunk = entire_dataset[chunk_start:chunk_end]

            results = {}
            
            # Create a shared queue for this chunk
            result_queue = mp.Manager().Queue()

            # Process scenes in the current chunk
            processes = []
            for scene_data in data_chunk:
                p = pool.apply_async(process_scene_queue, (scene_data, nusc, model, result_queue))
                processes.append(p)

            # Wait for all processes in this chunk to complete
            for p in processes:
                p.wait()

            # Collect results for this chunk
            while not result_queue.empty():
                result = result_queue.get()
                if result is not None:
                    sample_token, scene_results = result
                    results[sample_token] = scene_results

            # Save results for this chunk
            save_results(results, output_file_pkl)

            # Clear results and free memory
            results.clear()

    print("Processing completed.")

if __name__ == "__main__":
    main()