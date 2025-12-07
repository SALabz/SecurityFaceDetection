import os
import glob
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



#Authorized people
DATA_ROOT = r"data/lfw-deepfunneled"
#Custom faces       
CUSTOM_INTRUDER_ROOT = r"custom_test"         
DB_PATH = "authorized_embeddings.pkl"

THRESHOLD_SWEEP = np.linspace(0.4, 1.4, 50)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")
mtcnn = MTCNN(image_size=160, margin=20, device=device)
embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

with open(DB_PATH, "rb") as f:
    database = pickle.load(f)

print(f"The number of loaded authorized identities: {len(database)}\n")

def get_embedding(img_path):
    if not os.path.isfile(img_path):
        return None
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return None
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder(face)
    return emb.cpu().numpy().squeeze(0)


def compute_min_dist(emb, database):
    min_dist = float("inf")
    for db_emb in database.values():
        dist = np.linalg.norm(db_emb - emb)
        if dist < min_dist:
            min_dist = dist
    return min_dist


authorized_distances = []
for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        if file.lower().endswith(IMG_EXTENSIONS):
            path = os.path.join(root, file)
            emb = get_embedding(path)
            if emb is None:
                continue
            dist = compute_min_dist(emb, database)
            authorized_distances.append(dist)

print(f"Authorized test samples: {len(authorized_distances)}")


intruder_distances = []
for file in os.listdir(CUSTOM_INTRUDER_ROOT):
    if file.lower().endswith(IMG_EXTENSIONS):
        path = os.path.join(CUSTOM_INTRUDER_ROOT, file)
        emb = get_embedding(path)
        if emb is None:
            continue
        dist = compute_min_dist(emb, database)
        intruder_distances.append(dist)

print(f"Custom intruder test samples: {len(intruder_distances)}\n")


FAR = []   #Intruders that were incorrectly accepted
FRR = []   #Authorized users that were incorrectly rejected
TAR = []   #1 - FRR

authorized_distances = np.array(authorized_distances)
intruder_distances = np.array(intruder_distances)

for t in THRESHOLD_SWEEP:
    FAR.append(np.mean(intruder_distances < t))
    FRR.append(np.mean(authorized_distances > t))
    TAR.append(1 - FRR[-1])

FAR = np.array(FAR)
FRR = np.array(FRR)
TAR = np.array(TAR)


eer_idx = np.argmin(np.abs(FAR - FRR))
EER = (FAR[eer_idx] + FRR[eer_idx]) / 2
best_threshold = THRESHOLD_SWEEP[eer_idx]

print(f" Equal Error Rate (EER): {EER}")
print(f" Best Threshold: {best_threshold}")


#ROC curve plot
plt.plot(FAR, TAR)
plt.xlabel("False Acceptance Rate (FAR)")
plt.ylabel("True Acceptance Rate (TAR)")
plt.title("ROC Curve - Face Recognition Security")
plt.grid(True)
plt.show()


y_true = ([1] * len(authorized_distances) + [0] * len(intruder_distances))
y_pred = ([1 if d < best_threshold else 0 for d in authorized_distances] + [1 if d < best_threshold else 0 for d in intruder_distances])

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (Authorized=1, Intruder=0):")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

def test_single_image(img_path, threshold=best_threshold):
    emb = get_embedding(img_path)
    if emb is None:
        print("No face has been detected.")
        return

    dist = compute_min_dist(emb, database)
    print(f"Distance = {dist}")
    if dist < threshold:
        print("Authorized")
    else:
        print("Intruder")