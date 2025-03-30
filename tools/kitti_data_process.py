import numpy as np
import pickle
import argparse
from tqdm import tqdm

def convert_classes_kitti_infos_file(input_path, output_path, cls_mapping):
    """
    Convert class names in KITTI dataset and filter annotations.
    
    Args:
        input_path (str): Path to the source kitti_infos_train.pkl file.
        output_path (str): Path where the processed file will be saved.
        cls_mapping (dict): Mapping from original class names to new class names.
                            e.g., {'Car': 'car', 'Pedestrian': 'pedestrian'}
    """
    print(f"Loading KITTI info from: {input_path}")
    with open(input_path, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"Total samples before processing: {len(infos)}")
    total_annotations = 0
    total_kept = 0
    
    for info in tqdm(infos):
        
        annos = info["annos"]
        names = annos["name"]
        total_annotations += len(names)
        
        # Create mask for valid names (those in the mapping)
        keep_mask = np.array([name in cls_mapping.keys() for name in names], dtype=bool)
        
        # Map names using the provided mapping
        mapped_names = np.array([cls_mapping[name] for name in names[keep_mask]])
        
        # Update all annotation fields according to the mask
        for key in annos.keys():
            if key == "name":
                annos[key] = mapped_names
            else:
                if isinstance(annos[key], np.ndarray):
                    annos[key] = annos[key][keep_mask]
                elif isinstance(annos[key], list):
                    annos[key] = [item for i, item in enumerate(annos[key]) if keep_mask[i]]
        
        # Verify that all fields have the same length after filtering
        for key in annos.keys():
            if isinstance(annos[key], (np.ndarray, list)):
                assert len(annos[key]) == len(mapped_names), f"Length mismatch for {key}: {len(annos[key])} != {len(mapped_names)}"
        
        total_kept += len(mapped_names)
    
    print(f"Total annotations before processing: {total_annotations}")
    print(f"Total annotations after processing: {total_kept}")
    print(f"Removed annotations: {total_annotations - total_kept}")
    
    # Save processed data
    print(f"Saving processed KITTI info to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(infos, f)
    print("Processing completed successfully!")


def convert_classes_kitti_dbinfos_file(input_path, output_path, cls_mapping):
    print(f"Loading KITTI database info from: {input_path}")
    with open(input_path, 'rb') as f:
        dbinfos = pickle.load(f)
    ori_info_len = sum([len(v) for v in dbinfos.values()])
    print(f"Total samples before processing: {ori_info_len}")
    
    new_dbinfos = {}
    for cls_name, data in tqdm(dbinfos.items()):
        if cls_name in cls_mapping:
            new_cls_name = cls_mapping[cls_name]
            if new_cls_name not in new_dbinfos:
                new_dbinfos[new_cls_name] = data
            else:
                new_dbinfos[new_cls_name].extend(data)
    
    new_info_len = sum([len(v) for v in new_dbinfos.values()])
    print(f"Total samples after processing: {new_info_len}")
    print(f"Removed samples: {ori_info_len - new_info_len}")
    with open(output_path, 'wb') as f:
        pickle.dump(new_dbinfos, f)
    print(f"Saving processed KITTI database info to: {output_path}")

def main():
    """
    Main function to parse command line arguments and run the conversion.
    """
    parser = argparse.ArgumentParser(description="Convert class names in KITTI dataset annotations")
    parser.add_argument("--task", type=str, required=True, choices=["infos", "dbinfos"], help="Task to perform the conversion on")
    parser.add_argument("--input", type=str, required=True, help="Path to the source kitti_infos_train.pkl file")
    parser.add_argument("--output", type=str, required=True, help="Path where the processed file will be saved")
    args = parser.parse_args()
    
    # Define class mapping
    cls_mapping = {
        'Car': 'Car',
        'Truck': 'Truck',
        'Van': 'Truck',
        'Bus': 'Truck',
        'Pedestrian': 'Pedestrian',
        'Cyclist': 'Cyclist',
        'Motorcyclist': 'Cyclist',
        # Add more mappings as needed
    }
    if args.task == "infos":
        convert_classes_kitti_infos_file(args.input, args.output, cls_mapping)
    elif args.task == "dbinfos":
        convert_classes_kitti_dbinfos_file(args.input, args.output, cls_mapping)
    else:
        print(f"Invalid task {args.task}")

if __name__ == "__main__":
    main()
