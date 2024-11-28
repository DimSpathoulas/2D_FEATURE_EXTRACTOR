import argparse
import copy
import os
import os.path as osp
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from mrcnn import my_model_v3 as modellib
from mrcnn import visualize
from mrcnn.config import Config
from nuscenes import NuScenes
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from shapely.geometry import Polygon, LineString
from tqdm import tqdm

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
    transformed_corners = transformed_corners.T
    center_x = np.mean(transformed_corners[:, 0])
    center_y = np.mean(transformed_corners[:, 1])
    center_z = np.mean(transformed_corners[:, 2])

    dx = transformed_corners[0, 0] - transformed_corners[1, 0]
    dy = transformed_corners[0, 1] - transformed_corners[1, 1]
    rot_z = np.arctan2(dy, dx)

    return np.array([center_x, center_y, center_z, rot_z])


def initialize_model(model_dir=''):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


# def detect_objects(model, image):
#     r = model.detect([image], verbose=0)[0]  # h 1 den jero akoma
#     # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#     return r


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
        intersection_coords = None

        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        elif isinstance(img_intersection, LineString):
            intersection_coords = np.array([coord for coord in img_intersection.coords])

        if intersection_coords is not None:
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
            return min_x, min_y, max_x, max_y

    return None


def lidar_to_world(nusc, lidar_data, box):
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    box[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix, box[:3, :])
    box[:3, :] += np.array(cs_record['translation'])[:, None]

    poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    box[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix, box[:3, :])
    box[:3, :] += np.array(poserecord['translation'])[:, None]

    return box


def world_to_cam(nusc, cam, box):
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    box[:3, :] -= np.array(poserecord['translation'])[:, None]
    box[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, box[:3, :])

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    box[:3, :] -= np.array(cs_record['translation'])[:, None]
    box[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, box[:3, :])

    points = box[:3, :]
    depths = points[2, :]

    viewpad = np.eye(4)
    viewpad[:np.array(cs_record['camera_intrinsic']).shape[0], :np.array(cs_record['camera_intrinsic']).shape[1]] = np.array(cs_record['camera_intrinsic'])

    points = np.concatenate((points, np.ones((1, points.shape[1]))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    points = points / points[2:3, :].repeat(3, 0).reshape(3, points.shape[1])

    return points, depths


def save_results(new_data, base_filename):
    if os.path.exists(base_filename):
        with open(base_filename, 'rb') as read_file:
            existing_data = pickle.load(read_file)
        if not isinstance(existing_data, dict):
            existing_data = {}
    else:
        existing_data = {}

    existing_data.update(new_data)

    with open(base_filename, 'wb') as write_file:
        pickle.dump(existing_data, write_file)


def update_or_append_result(results_temp, track_name, final_box_pred, Projection, point_cloud_feats, feature_vector,
                            camera_onehot_vec, pred_score, sample_token, timestamp):
    for existing_result in results_temp[track_name]:
        if np.array_equal(existing_result['box'], final_box_pred):
            old_area = np.prod(existing_result['projection'][2:] - existing_result['projection'][:2])
            new_area = np.prod(Projection[2:] - Projection[:2])
            if new_area > old_area:
                existing_result.update({
                    'projection': Projection,
                    'point_cloud_features': point_cloud_feats,
                    'feature_vector': feature_vector,
                    'camera_onehot_vector': camera_onehot_vec,
                    'pred_score': pred_score
                })
            return
    
    results_temp[track_name].append({
        'sample_token': sample_token,
        'timestamp': timestamp,
        'track_name': track_name,
        'box': final_box_pred,
        'projection': Projection,
        'point_cloud_features': point_cloud_feats,
        'feature_vector': feature_vector,
        'camera_onehot_vector': camera_onehot_vec,
        'pred_score': pred_score
    })

    
def process_detection(v, step, jjj, projections, data, i, nusc, lidar_data, feature_vectors,
                      camera_onehot_vec, results_temp, sample_token, timestamp):
    Projection = projections[step]
    label = data['pred_labels'][jjj].cpu().numpy()[0]
    track_name = detector_classes[label]

    point_cloud_feats = data['features'][jjj].cpu().numpy().reshape(512, 3, 3)
    pred_score = data['pred_scores'][jjj].cpu().numpy()
    pred_boxes = data['pred_boxes'][jjj].cpu().numpy().reshape(9)

    lidar_world = create_box(pred_boxes).T
    world_box = lidar_to_world(nusc, lidar_data, lidar_world)
    pred_box_world = retrieve_box_info(world_box)

    final_box_pred = np.array([*pred_box_world[:3], *pred_boxes[3:6], pred_box_world[3], *pred_boxes[7:]])

    update_or_append_result(results_temp, track_name, final_box_pred, Projection, point_cloud_feats,
                            feature_vectors[step], camera_onehot_vec, pred_score, sample_token, timestamp)


def initialize_data(nusc, sample_token):
    sample_data = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample_data['data']['LIDAR_TOP'])
    camera_vector = {}

    for v, sensor in enumerate(sensors):
        cam_data = nusc.get('sample_data', sample_data['data'][sensor])
        im = Image.open(osp.join(nusc.dataroot, cam_data['filename']))
        opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        camera_vector[v] = {
            'image': opencvImage,
            'cam': cam_data
        }

    return lidar_data, camera_vector

def prepare_sample_data(args):
    sample_token, nusc = args
    sample = nusc.get('sample', sample_token)
    lidar_data, camera_vector = initialize_data(nusc, sample_token)
    projections, _ = prepare_projections_parallel(camera_vector, sample['num_labels'], sample, 0, nusc, lidar_data)
    return sample_token, projections


def prepare_projections_for_camera(v, cam_info, num_objects, data, i, nusc, lidar_data):
    projection_list = []
    info_stack = []

    for j in range(num_objects):
        label = data['pred_labels'][j].cpu().numpy()
        track_name = detector_classes[label[0]]
        if track_name not in tracking_names:
            continue

        pred_box = data['pred_boxes'][j].cpu().numpy()
        box = create_box(pred_box).T
        box = lidar_to_world(nusc, lidar_data, box)
        pred_box_worlds = retrieve_box_info(transformed_corners=box)

        final_box_preds = np.array([*pred_box_worlds[:3], *pred_box[3:6], pred_box_worlds[3], *pred_box[7:]])

        final_box = create_box(final_box_preds).T
        points, depths = world_to_cam(nusc, cam_info['cam'], final_box)

        mask = depths > 2.0
        points = points[:2, mask].T

        if points.shape[0] <= 2:
            continue

        final_coords = post_process_coords(points)
        if final_coords is None:
            continue

        min_x, min_y, max_x, max_y = np.clip(final_coords.astype(int), [4, 4, 0, 0], [1596, 896, 1596, 896])

        projection = np.array([min_y, min_x, max_y, max_x])
        projection_list.append(projection)
        info_stack.append(j)

    return projection_list, info_stack



def prepare_projections_parallel(camera_vector, num_objects, data, i, nusc, lidar_data):
    projection_dict = {}
    dict_info_stack = {}

    for v, cam_info in camera_vector.items():
        projection_list, info_stack = prepare_projections_for_camera(
            v, cam_info, num_objects, data, i, nusc, lidar_data)
        projection_dict[v] = projection_list
        dict_info_stack[v] = info_stack

    return projection_dict, dict_info_stack



def process_camera_data_for_view(args):
    v, cam_info, projections, jj, model, data, i, nusc, lidar_data, sample_token, timestamp = args
    camera_onehot_vec = np.zeros(len(sensors))
    camera_onehot_vec[v] = 1

    image = cam_info['image']
    mrcnn_results = model.detect([image], projections=projections, verbose=0)

    feature_vectors = mrcnn_results

    results_temp = defaultdict(list)
    for step, jjj in enumerate(jj):
        process_detection(v, step, jjj, projections, data, i, nusc, lidar_data, feature_vectors,
                          camera_onehot_vec, results_temp, sample_token, timestamp)

    return results_temp



def process_camera_data_parallel(camera_vector, projection_dict, dict_info_stack, model, data, i,
                                 nusc, lidar_data, sample_token, timestamp):
    with Pool(processes=1) as pool:
        args = [(v, cam_info, projection_dict[v], dict_info_stack[v], model, data, i, nusc, lidar_data, sample_token, timestamp)
                for v, cam_info in camera_vector.items()]
        results = pool.map(process_camera_data_for_view, args)

    combined_results = defaultdict(list)
    for result in results:
        for track_name, detections in result.items():
            combined_results[track_name].extend(detections)

    return combined_results


def process_sample(sample, projections, model, nusc):
    results_temp = defaultdict(list)
    
    for v, cam_projections in projections.items():
        camera_onehot_vec = np.zeros(len(sensors))
        camera_onehot_vec[v] = 1

        mrcnn_results = model.detect([sample['image']], projections=cam_projections, verbose=0)
        feature_vectors = mrcnn_results

        for step, projection in enumerate(cam_projections):
            process_detection(v, step, step, [projection], sample, 0, nusc, sample['lidar_data'], 
                              feature_vectors, camera_onehot_vec, results_temp, 
                              sample['metadata'][0]['token'], sample['metadata'][0]['timestamp'])

    return results_temp



def load_data(file_path, batch_size=100):
    """
    Generator function to load data in batches
    """
    data = np.load(file_path, allow_pickle=True)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]



def process_data(data_generator, nusc, model, num_workers, batch_size=10):
    results = {}
    
    for large_batch in tqdm(data_generator, desc="Processing large batches"):
        for i in range(0, len(large_batch), batch_size):
            batch = large_batch[i:i+batch_size]
            
            prepare_args = [(sample['metadata'][0]['token'], nusc) for sample in batch]

            with Pool(processes=num_workers) as pool:
                prepared_data = list(pool.map(prepare_sample_data, prepare_args))

            batch_results = {}
            for sample_token, projections in prepared_data:
                sample_data = next(sample for sample in batch if sample['metadata'][0]['token'] == sample_token)
                sample_results = process_sample(sample_data, projections, model, nusc)
                batch_results[sample_token] = sample_results

            results.update(batch_results)

            if len(results) % 500 == 0:
                save_results(results, f"temp_results_{len(results)}.pkl")
                results = {}

    if results:
        save_results(results, "final_results.pkl")

    return results



def main():
    parser = argparse.ArgumentParser(description="Project 3d detections to camera planes and extract feature vectors.")

    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version --v1.0-trainval or v1.0-mini')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--detection_file', type=str,
                        default="/home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_train_2.npy",
                        help='Path to the npy detection file')
    parser.add_argument('--output_file', type=str,
                        default='mrcnn_train_2_optimized.pkl',
                        help='Path to the output pkl file')
    parser.add_argument('--num_workers', type=int, default=cpu_count() - 2,
                        help='Number of worker processes for CPU tasks')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Batch size for processing')
    parser.add_argument('--load_batch_size', type=int, default=250,
                        help='Batch size for loading data from disk')

    args = parser.parse_args()

    # Configure TensorFlow to use GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    data_generator = load_data(args.detection_file, args.load_batch_size)
    
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    model = initialize_model(model_dir='lgs')

    output_file_pkl = args.output_file
    print(f"Number of workers: {args.num_workers}")
    results = process_data(data_generator, nusc, model, args.num_workers, batch_size=args.batch_size)
    save_results(results, output_file_pkl)

if __name__ == "__main__":
    main()
