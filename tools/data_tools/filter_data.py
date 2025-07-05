# data/DAIR-V2X/V2X-Seq-SPD/cooperative/data_info.json 
# {"vehicle_frame": "000009", "infrastructure_frame": "000000", "vehicle_sequence": "0000", "infrastructure_sequence": "0000", 
#  "system_error_offset": {"delta_x": -0.7, "delta_y": -0.13999999999999996}}

# /home/dzh/jlk/DAIR-V2X/data/DAIR-V2X/V2X-Seq-SPD/vehicle-side/data_info.json
# {"image_path": "image/000000.jpg", "pointcloud_path": "velodyne/000000.pcd", "calib_camera_intrinsic_path": "calib/camera_intrinsic/000000.json", 
#  "calib_lidar_to_camera_path": "calib/lidar_to_camera/000000.json", "calib_lidar_to_novatel_path": "calib/lidar_to_novatel/000000.json", 
#  "calib_novatel_to_world_path": "calib/novatel_to_world/000000.json", "label_camera_std_path": "label/camera/000000.json", "label_lidar_std_path": "label/lidar/000000.json", 
#  "intersection_loc": "", "image_timestamp": "1626155123061000", "pointcloud_timestamp": "1626155122981522", "frame_id": "000000", "start_frame_id": "000000", 
#  "end_frame_id": "000209", "num_frames": 210, "sequence_id": "0000"}

# /home/dzh/jlk/DAIR-V2X/data/DAIR-V2X/V2X-Seq-SPD/cooperative/label/000009.json
# {"token": "46870de4-7673-3fee-acbc-ab4b35f70834", "type": "Car", "track_id": "002762", "truncated_state": 0, "occluded_state": 2, "alpha": -1.364216, 
#  "2d_box": {"xmin": 865.511414, "ymin": 567.811646, "xmax": 928.400513, "ymax": 615.924256}, "3d_dimensions": {"l": 4.251301, "w": 1.8234, "h": 1.570357}, 
#  "3d_location": {"x": 27.07514, "y": 4.712578, "z": -0.922006}, "rotation": -0.034323, "from_side": "veh", "veh_pointcloud_timestamp": "1626155123881356", 
#  "inf_pointcloud_timestamp": "1626155123944230", "veh_frame_id": "000009", "inf_frame_id": "000000", "veh_track_id": "009525", "inf_track_id": "-1", 
#  "veh_token": "46870de4-7673-3fee-acbc-ab4b35f70834", "inf_token": "-1"}

mapping = {
    'Car': 'Car',
    'Truck': 'Truck',
    'Van': 'Truck',
    'Bus': 'Truck',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Motorcyclist': 'Cyclist',
}

import json
import os
import pandas as pd
with open('/mnt/data4/jlk/V2X-Seq-SPD/vic3d-early-fusion-training/data_info.json', 'r') as f:
    coor_json = json.loads(f.read())
with open('/mnt/data5/fjh/PillarNeXt/codebase/pillarnext/tools/data_tools/output/point_in_boxes_filtered.json', 'r') as f:
    points_json = json.loads(f.read())
with open('/mnt/data5/fjh/PillarNeXt/codebase/pillarnext/tools/data_tools/output/labels.json', 'r') as f:
    labels_json = json.loads(f.read())

base_path = '/mnt/data4/jlk/V2X-Seq-SPD/vic3d-early-fusion-training/label/lidar'
save_path = '/mnt/data5/fjh/PillarNeXt/codebase/pillarnext/tools/data_tools/output'
split = {}
new_id = "0000"
topk = 3

def inc(xx):
    return "{:0>4}".format(int(xx) + 1)

las = -1
df = []

for item in coor_json:
    frame = item['frame_id']
    id = item['sequence_id']
    if id != las or dic['object_count'] > 500:
        new_id = inc(new_id)
        split[new_id] = []
        dic = {'id': new_id, 'sequence': id, 'start_frame': frame, 'end_frame': frame, 'frame_count': 0, 'point_in_boxes': 0, 'object_count': 0, 'Car': 0, 'Truck': 0, 'Pedestrian': 0, 'Cyclist': 0}
        df.append(dic)
        
        
    las = id

    split[new_id].append(frame)
    
    # cnt[id] += sum(1 for item in label_json if item['type'] == 'Pedestrian')
    dic['end_frame'] = frame
    dic['frame_count'] += 1
    dic['point_in_boxes'] += points_json[frame]
    for key, value in labels_json[frame].items():
        if key not in mapping:
            continue
        dic[mapping[key]] += value
        dic['object_count'] += value

new_df = []
for item in df:
    if item['object_count'] < 500 or item['Pedestrian'] == 0 or item['Cyclist'] == 0 or item['Truck'] == 0:
        split.pop(item['id'])
        # print(item['id'])
    else:
        new_df.append(item)

selected_df = []
df = sorted(new_df, key=lambda x: x['point_in_boxes'], reverse=True)
cnt = {}
for item in df:
    if cnt.get(item['sequence'], 0) >= topk: 
        split.pop(item['id'])
        # print(item['id'])
        continue
    cnt[item['sequence']] = cnt.get(item['sequence'], 0) + 1
    selected_df.append(item)

print(len(split))
# print(split)
with open(os.path.join(save_path, 'my_split_spd_final.json'), 'w') as f:
    f.write(json.dumps(split))
frames = []
for key, value in split.items():
    for item in value:
        frames.append(item)
with open(os.path.join(save_path, 'kitti.txt'), 'w') as f:
    f.write('{}\n'.format('\n'.join(frames)))

df = pd.DataFrame(selected_df)
df.to_csv(os.path.join(save_path, 'result_500.csv'), index=False)