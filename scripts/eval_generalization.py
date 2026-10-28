import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from models.detection import PersonDetector

resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
modules = list(resnet.children())[:-1]
cnn = torch.nn.Sequential(*modules).eval()

t_global = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((128, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

t_part = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((64, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_universal_sig(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    h, w = crop.shape[:2]
    
    # 1. Global CNN
    rgb_g = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        v_g = cnn(t_global(rgb_g).unsqueeze(0)).flatten().numpy()
    v_g = v_g / (np.linalg.norm(v_g) + 1e-7)

    # 2. Overlapping parts
    slices = [
        crop[0:int(h * 0.25), :],
        crop[int(h * 0.20):int(h * 0.65), :],
        crop[int(h * 0.55):h, :]
    ]
    part_cnns, part_colors, part_textures = [], [], []
    for s in slices:
        if s.size == 0 or s.shape[0] < 4 or s.shape[1] < 4:
            part_cnns.append(np.zeros(512))
            part_colors.append(np.zeros(48 + 32))
            part_textures.append(np.zeros(16))
            continue
        
        # Deep Part CNN
        rgb_s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            v_p = cnn(t_part(rgb_s).unsqueeze(0)).flatten().numpy()
        part_cnns.append(v_p / (np.linalg.norm(v_p) + 1e-7))
        
        # Comprehensive Color: Lab (48 bins) + HSV (32 bins)
        lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
        h_L = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
        h_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
        h_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
        lab_hist = np.concatenate([h_L, h_a, h_b])
        lab_hist = lab_hist / (np.linalg.norm(lab_hist) + 1e-7)
        
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        h_H = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        h_S = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        h_V = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hsv_hist = np.concatenate([h_H, h_S, h_V])
        hsv_hist = hsv_hist / (np.linalg.norm(hsv_hist) + 1e-7)
        
        col = np.concatenate([lab_hist, hsv_hist])
        part_colors.append(col / (np.linalg.norm(col) + 1e-7))
        
        # Texture: gradient magnitude
        gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        tex_hist = cv2.calcHist([mag], [0], None, [16], [0, 256]).flatten()
        part_textures.append(tex_hist / (np.linalg.norm(tex_hist) + 1e-7))
        
    return v_g, part_cnns, part_colors, part_textures

def compare_universal(sig1, sig2):
    if sig1 is None or sig2 is None:
        return 0.0
    vg1, cnn1, col1, tex1 = sig1
    vg2, cnn2, col2, tex2 = sig2
    
    # Global CNN match (30%)
    s_global = max(0.0, float(np.dot(vg1, vg2)))
    
    # Part weights: Head (15%), Torso (35%), Lower (20%)
    pw = [0.15, 0.35, 0.20]
    s_parts = 0.0
    for i in range(3):
        c_sim = max(0.0, float(np.dot(cnn1[i], cnn2[i])))
        col_sim = max(0.0, float(np.dot(col1[i], col2[i])))
        tex_sim = max(0.0, float(np.dot(tex1[i], tex2[i])))
        score = 0.50 * c_sim + 0.40 * col_sim + 0.10 * tex_sim
        s_parts += pw[i] * score
        
    total = 0.30 * s_global + s_parts
    return float(np.clip(total, 0.0, 1.0))

p_group = r'C:\Users\dhira\.gemini\antigravity-ide\brain\3ac8cc61-8bbf-46ee-84a6-f20dad9c5ac8\.user_uploaded\media_1793012425936.jpg'
group = cv2.imread(p_group)
det = PersonDetector('yolov8n.pt', conf_thresh=0.20)
res = det.detect(group)
boxes = res.boxes.xyxy.cpu().numpy()

cand_crops = [group[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
cand_sigs = [extract_universal_sig(c) for c in cand_crops]

test_indices = [0, 1, 2, 6, 8, 9]
labels = {
    0: 'Person 0 (Leftmost, dark outfit)',
    1: 'Person 1 (Left-center, light shirt)',
    2: 'Person 2 (Right, white/cream shirt)',
    6: 'Person 6 (Center, black shirt)',
    8: 'Person 8 (Right, mustard/tan shirt)',
    9: 'Person 9 (Center, blue shirt)'
}

print('=== TESTING GENERALIZATION ACROSS DIFFERENT PEOPLE ===')
for idx in test_indices:
    query_crop = cand_crops[idx]
    query_sig = cand_sigs[idx]
    
    scores = []
    for j, csig in enumerate(cand_sigs):
        score = compare_universal(query_sig, csig)
        scores.append((j, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_j, top_s = scores[0]
    second_j, second_s = scores[1]
    is_correct = (top_j == idx)
    status = 'CORRECT (Rank 1)' if is_correct else f'WRONG (Got Person {top_j})'
    print(f'{labels[idx]}: {status}')
    print(f'   Top #1 score: {top_s*100:.1f}%, Runner-up (Person {second_j}) score: {second_s*100:.1f}%, Margin: {(top_s - second_s)*100:.1f}%')

p_single = r'C:\Users\dhira\.gemini\antigravity-ide\brain\3ac8cc61-8bbf-46ee-84a6-f20dad9c5ac8\.user_uploaded\media_1793012434352.jpg'
single = cv2.imread(p_single)
s_res = det.detect(single)
sb = s_res.boxes.xyxy[0].cpu().numpy()
single_crop = single[int(sb[1]):int(sb[3]), int(sb[0]):int(sb[2])]
single_sig = extract_universal_sig(single_crop)

single_scores = [(j, compare_universal(single_sig, csig)) for j, csig in enumerate(cand_sigs)]
single_scores.sort(key=lambda x: x[1], reverse=True)
print('\n=== TESTING USER EXTERNAL QUERY IMAGE ===')
print('Top 3 matches for external query image:')
for rank, (j, s) in enumerate(single_scores[:3], 1):
    label = labels.get(j, f'Person {j}')
    print(f'Rank #{rank}: Person {j} ({label}) - Score: {s*100:.1f}%')
