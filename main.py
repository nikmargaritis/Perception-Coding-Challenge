import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# CONFIGURATION
# ==========================================

DATASETS = [
    {
        "name": "final_smooth_sequence",
        "csv": "dataset/bbox_light.csv",
        "npz": "dataset/xyz",
        "rgb": "dataset/rgb",
        "smoothing_window": 15,   
        "max_jump": 5.0          # INCREASED to allow pedestrian movement
    }
]

# ==========================================
# UTILITIES
# ==========================================

def get_clean_position(npz_path, center_u, center_v, patch_size=2):
    try:
        data = np.load(npz_path)
        points = data["xyz"] 
    except Exception:
        return None, "Load Error"

    if len(points.shape) != 3:
        return None, f"Bad Shape {points.shape}"
        
    H, W, C = points.shape
    u_min, u_max = max(0, center_u - patch_size), min(W, center_u + patch_size + 1)
    v_min, v_max = max(0, center_v - patch_size), min(H, center_v + patch_size + 1)
    
    patch = points[v_min:v_max, u_min:u_max, :]
    if patch.shape[-1] > 3:
        patch = patch[:, :, :3]
    
    flat_points = patch.reshape(-1, 3)
    valid_mask = np.isfinite(flat_points).all(axis=1)
    clean_points = flat_points[valid_mask]
    non_zero_mask = ~np.all(clean_points == 0, axis=1)
    clean_points = clean_points[non_zero_mask]
    
    if len(clean_points) == 0:
        return None, "All-Zero/Noise"
        
    return np.median(clean_points, axis=0), "Success"

def fig_to_img(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

def fix_csv_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        'xmin': 'x_min', 'left': 'x_min', 'x1': 'x_min',
        'ymin': 'y_min', 'top': 'y_min',  'y1': 'y_min',
        'xmax': 'x_max', 'right': 'x_max', 'x2': 'x_max',
        'ymax': 'y_max', 'bottom': 'y_max', 'y2': 'y_max',
        'frame': 'frame_id', 'id': 'frame_id'
    }
    new_names = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=new_names, inplace=True)
    return df

def get_smooth_path(raw_points, window_size):
    """ Helper to smooth a list of [x, y] points using Pandas """
    if not raw_points: return []
    
    cleaned_points = []
    for p in raw_points:
        if p is None:
            cleaned_points.append([np.nan, np.nan])
        else:
            cleaned_points.append(p)
            
    df = pd.DataFrame(cleaned_points, columns=['x', 'y'])
    df_filled = df.ffill().bfill() 
    df_smooth = df_filled.rolling(window=window_size, min_periods=1, center=True).mean()
    return df_smooth.values

# ==========================================
# MAIN LOGIC
# ==========================================

def process_sequence(dataset):
    name = dataset['name']
    print(f"\n--- Processing: {name} ---")
    
    # 1. Load Data
    if not os.path.exists(dataset['csv']):
        print(f"Error: CSV not found at {dataset['csv']}")
        return
    df = pd.read_csv(dataset['csv'])
    df = fix_csv_columns(df)

    # 2. Interactive Tracker Setup
    first_row = df.iloc[0]
    fid_int = int(float(first_row['frame_id']))
    img_path = os.path.join(dataset['rgb'], f"left{str(fid_int).zfill(6)}.png") 
    if not os.path.exists(img_path): img_path = os.path.join(dataset['rgb'], f"frame_{str(fid_int).zfill(4)}.png")
    
    frame0 = cv2.imread(img_path)
    if frame0 is None:
        print("Error: Could not load first frame for setup.")
        return

    trackers = []
    tracker_labels = []
    tracker_colors = []
    
    print("\n--- INTERACTIVE SETUP ---")
    print("1. Draw box -> SPACE.")
    print("2. Type label (e.g. 'Pedestrian', 'Cart').")
    print("3. ESC when done.")
    
    while True:
        bbox = cv2.selectROI("Select Objects", frame0, False)
        cv2.destroyWindow("Select Objects")
        if bbox == (0,0,0,0): break
        if bbox[2] == 0 or bbox[3] == 0: continue 
        
        try: tracker = cv2.TrackerCSRT_create()
        except: tracker = cv2.TrackerKCF_create()
            
        tracker.init(frame0, bbox)
        trackers.append(tracker)
        
        lbl = input(f"Object {len(trackers)} Label: ")
        tracker_labels.append(lbl)
        
        if 'cart' in lbl.lower(): tracker_colors.append((255, 0, 255)) 
        elif 'pedestrian' in lbl.lower(): tracker_colors.append((0, 255, 255)) 
        elif 'barrel' in lbl.lower(): tracker_colors.append((0, 165, 255))
        else: tracker_colors.append((255, 255, 0)) 
        
    cv2.destroyAllWindows()

    # 3. First Pass: Tracking
    print(f"Scanning {len(df)} frames...")
    
    ego_raw = []
    obj_raw = {i: [] for i in range(len(trackers))}
    obj_boxes = {i: [] for i in range(len(trackers))} 
    
    R_matrix = None

    for idx, row in df.iterrows():
        try: fid_int = int(float(row['frame_id']))
        except: continue
        fid_6 = str(fid_int).zfill(6)
        fid_4 = str(fid_int).zfill(4)
        
        npz_file = None
        for cand in [f"depth{fid_6}.npz", f"frame_{fid_4}.npz", f"{fid_6}.npz", f"{fid_4}.npz"]:
            p = os.path.join(dataset['npz'], cand)
            if os.path.exists(p): 
                npz_file = p
                break
        
        img_file = None
        for cand in [f"left{fid_6}.png", f"frame_{fid_4}.png", f"{fid_6}.png", f"{fid_4}.png"]:
            p = os.path.join(dataset['rgb'], cand)
            if os.path.exists(p):
                img_file = p
                break
        
        if not npz_file or not img_file: continue
        frame = cv2.imread(img_file)
        
        # --- A. Track Ego ---
        u = int((row['x_min'] + row['x_max']) / 2)
        v = int((row['y_min'] + row['y_max']) / 2)
        p_cam, _ = get_clean_position(npz_file, u, v)
        
        current_ego = None
        if p_cam is not None:
            if R_matrix is None:
                angle = -np.arctan2(p_cam[1], p_cam[0])
                R_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            
            vec_rot = R_matrix @ np.array([-p_cam[0], -p_cam[1]])
            current_ego = vec_rot
            ego_raw.append(current_ego)
        else:
            ego_raw.append(ego_raw[-1] if ego_raw else None)
            current_ego = ego_raw[-1] if ego_raw else np.array([0,0])

        # --- B. Track Objects ---
        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            obj_boxes[i].append(box if success else None)
            
            val_pos = None
            if success and current_ego is not None:
                tx, ty, tw, th = [int(x) for x in box]
                cx, cy = tx + tw//2, ty + th//2
                
                p_obj_cam, _ = get_clean_position(npz_file, cx, cy)
                
                if p_obj_cam is not None:
                    vec_obj_rot = R_matrix @ np.array([p_obj_cam[0], p_obj_cam[1]])
                    potential_pos = current_ego + vec_obj_rot
                    
                    last_valid = None
                    if obj_raw[i]: 
                        valid_hist = [p for p in obj_raw[i] if p is not None]
                        if valid_hist: last_valid = valid_hist[-1]
                    
                    if last_valid is not None:
                        dist = np.linalg.norm(potential_pos - last_valid)
                        if dist < dataset.get('max_jump', 5.0):
                            val_pos = potential_pos
                        else:
                            val_pos = last_valid 
                    else:
                        val_pos = potential_pos
            
            obj_raw[i].append(val_pos)

    # 4. Smoothing Phase
    print("Smoothing trajectories...")
    window = dataset.get('smoothing_window', 10)
    
    ego_smooth = get_smooth_path(ego_raw, window)
    
    obj_smooth = {}
    for i in range(len(trackers)):
        obj_smooth[i] = get_smooth_path(obj_raw[i], window)

    # 5. Rendering
    # UPDATED: Changed output paths
    video_path = "trajectory.mp4"
    image_path = "trajectory.png"
    print(f"Rendering to {video_path} and {image_path}...")
    
    all_x, all_y = [], []
    if len(ego_smooth) > 0:
        all_x.extend(ego_smooth[:, 0]); all_y.extend(ego_smooth[:, 1])
    for i in obj_smooth:
        if len(obj_smooth[i]) > 0:
            all_x.extend(obj_smooth[i][:, 0]); all_y.extend(obj_smooth[i][:, 1])
            
    if not all_x: all_x = [0]; all_y = [0]
    
    x_min, x_max = min(all_x)-5, max(all_x)+5
    y_min, y_max = min(all_y)-5, max(all_y)+5
    
    fig, ax = plt.subplots(figsize=(10, 5))
    video_writer = None
    fps = 10
    
    for idx in range(len(ego_smooth)):
        row = df.iloc[idx]
        fid_int = int(float(row['frame_id']))
        fid_6 = str(fid_int).zfill(6)
        fid_4 = str(fid_int).zfill(4)
        
        img_file = None
        for cand in [f"left{fid_6}.png", f"frame_{fid_4}.png", f"{fid_6}.png", f"{fid_4}.png"]:
            p = os.path.join(dataset['rgb'], cand)
            if os.path.exists(p):
                img_file = p
                break
        
        if not img_file: continue
        frame = cv2.imread(img_file)
        
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"BEV Trajectory")
        ax.grid(True)
        ax.axis('equal')
        
        # SHOW ACTIVE PATHS
        ax.plot(ego_smooth[:idx+1, 0], ego_smooth[:idx+1, 1], 'b-', alpha=0.8, label="Ego")
        ax.plot(ego_smooth[idx, 0], ego_smooth[idx, 1], 'bo', markersize=8)
        
        for i in obj_smooth:
            path = obj_smooth[i]
            if idx < len(path) and not np.isnan(path[idx]).any():
                color_norm = [c/255.0 for c in tracker_colors[i][::-1]]
                ax.plot(path[:idx+1, 0], path[:idx+1, 1], color=color_norm, linestyle='--')
                ax.scatter(path[idx, 0], path[idx, 1], color=color_norm, marker='s', s=60, label=tracker_labels[i])

        ax.scatter(0, 0, c='yellow', edgecolors='k', marker='*', s=300, label='Light', zorder=10)
                
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        
        # UPDATED: Save PNG at the final frame
        if idx == len(ego_smooth) - 1:
            plt.savefig(image_path)
            print(f"Static trajectory saved to {image_path}")
            
        map_img = fig_to_img(fig)
        
        for i in range(len(trackers)):
            if idx < len(obj_boxes[i]):
                box = obj_boxes[i][idx]
                if box is not None:
                    tx, ty, tw, th = [int(v) for v in box]
                    col = tracker_colors[i]
                    cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), col, 2)
                    
                    lbl_text = tracker_labels[i]
                    (text_w, text_h), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = ty - 10 if ty > 20 else ty + 20
                    cv2.rectangle(frame, (tx, label_y - text_h - 5), (tx + text_w, label_y + 5), (0,0,0), -1)
                    cv2.putText(frame, lbl_text, (tx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        h_cam, w_cam, _ = frame.shape
        h_map, w_map, _ = map_img.shape
        scale = h_cam / h_map
        map_resized = cv2.resize(map_img, (int(w_map * scale), h_cam))
        final_frame = np.hstack((frame, map_resized))
        
        if video_writer is None:
            h, w, _ = final_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
        video_writer.write(final_frame)
        print(f"  Frame {idx}/{len(ego_smooth)}", end='\r')

    if video_writer: video_writer.release()
    plt.close(fig)

if __name__ == "__main__":
    for ds in DATASETS:
        process_sequence(ds)