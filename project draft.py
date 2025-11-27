# # """
# # face_recognition_system.py
# # Run in VS Code with a Python interpreter (ideally with CUDA GPU, e.g. RTX 3090).

# # Requirements (install once in your environment):
# #     pip install torch torchvision facenet-pytorch pillow
# # """

# # import os
# # import sys
# # import glob
# # from typing import Dict, Tuple, Optional

# # import torch
# # from torch import Tensor
# # from facenet_pytorch import MTCNN, InceptionResnetV1
# # from PIL import Image


# # # ---------------------------
# # # CONFIG
# # # ---------------------------

# # # Folder of known people:
# # # Each subfolder = one person, images inside are that person's face photos.
# # DATA_ROOT = r"C:\Users\aldwa\OneDrive\Documents\achool\data\lfw-deepfunneled"


# # # Distance threshold for intruder detection (tune later)
# # INTRUDER_THRESHOLD = 0.9

# # # Image extensions to consider
# # IMG_EXTS = (".jpg", ".jpeg", ".png")


# # # ---------------------------
# # # DEVICE + MODELS
# # # ---------------------------

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")

# # mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
# # embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()


# # # ---------------------------
# # # EMBEDDING FUNCTIONS
# # # ---------------------------

# # def get_embedding(img_path: str) -> Optional[Tensor]:
# #     """
# #     Load image, detect face with MTCNN, run through FaceNet to get a 512-dim embedding.
# #     Returns a 1D Tensor on CPU (shape [512]) or None if no face detected.
# #     """
# #     try:
# #         img = Image.open(img_path).convert("RGB")
# #     except Exception as e:
# #         print(f"[WARN] Could not open {img_path}: {e}")
# #         return None

# #     face = mtcnn(img)  # tensor on `device` or None
# #     if face is None:
# #         print(f"[WARN] No face detected in {img_path}")
# #         return None

# #     if face.ndim == 3:
# #         face = face.unsqueeze(0)  # (1, C, H, W)

# #     face = face.to(device)

# #     with torch.no_grad():
# #         emb = embedder(face)  # (1, 512)

# #     return emb.squeeze(0).cpu()  # (512,)


# # def build_database(root: str) -> Dict[str, Tensor]:
# #     """
# #     Build dictionary: person_name -> mean embedding across that person's images.
# #     """
# #     if not os.path.isdir(root):
# #         print(f"[ERROR] DATA_ROOT does not exist or is not a directory: {root}")
# #         sys.exit(1)

# #     people = sorted(d for d in os.listdir(root)
# #                     if os.path.isdir(os.path.join(root, d)))

# #     if not people:
# #         print(f"[ERROR] No person folders found in {root}")
# #         sys.exit(1)

# #     database: Dict[str, Tensor] = {}

# #     print(f"Building database from: {root}")
# #     print(f"Found {len(people)} person folders")

# #     for person in people:
# #         person_dir = os.path.join(root, person)
# #         img_paths = []
# #         for ext in IMG_EXTS:
# #             img_paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))

# #         if not img_paths:
# #             print(f"[WARN] No images found for {person}, skipping")
# #             continue

# #         embs = []
# #         for p in img_paths:
# #             emb = get_embedding(p)
# #             if emb is not None:
# #                 embs.append(emb)

# #         if not embs:
# #             print(f"[WARN] No valid embeddings for {person}, skipping")
# #             continue

# #         person_emb = torch.stack(embs, dim=0).mean(dim=0)  # (512,)
# #         database[person] = person_emb
# #         print(f"  Loaded {person}: {len(embs)} images")

# #     print(f"Database built with {len(database)} people")
# #     return database


# # def identify_face(
# #     img_path: str,
# #     database: Dict[str, Tensor],
# #     threshold: float = INTRUDER_THRESHOLD
# # ) -> Tuple[str, float]:
# #     """
# #     Compare input face embedding to all known people.
# #     Returns (label, distance). Label is either a person's name or 'INTRUDER'.
# #     """
# #     emb = get_embedding(img_path)
# #     if emb is None:
# #         return "NO_FACE_DETECTED", float("inf")

# #     dists = []
# #     for name, ref_emb in database.items():
# #         dist = torch.norm(ref_emb - emb).item()
# #         dists.append((name, dist))

# #     if not dists:
# #         return "DATABASE_EMPTY", float("inf")

# #     best_name, best_dist = min(dists, key=lambda x: x[1])

# #     if best_dist > threshold:
# #         return "INTRUDER", best_dist
# #     else:
# #         return best_name, best_dist


# # # ---------------------------
# # # MAIN CLI
# # # ---------------------------

# # def main():
# #     print("=== Face Recognition / Intruder Detection (Embedding-Based) ===")
# #     print(f"DATA_ROOT = {DATA_ROOT}")
# #     db = build_database(DATA_ROOT)

# #     if not db:
# #         print("[ERROR] Database is empty. Check your DATA_ROOT.")
# #         return

# #     print("\nReady.")
# #     print("Type a path to an image file to test.")
# #     print("Type 'q' to quit.\n")

# #     while True:
# #         img_path = input("Image path> ").strip()
# #         if img_path.lower() in ("q", "quit", "exit"):
# #             break

# #         if not os.path.isfile(img_path):
# #             print("[ERROR] File does not exist.")
# #             continue

# #         label, dist = identify_face(img_path, db, INTRUDER_THRESHOLD)
# #         print(f"Result: {label}  (distance={dist:.4f})\n")


# # if __name__ == "__main__":
# #     main()


# #============================================================================

# """
# Simple CNN baseline on LFW (deep learning, from-scratch CNN)
# Train: multi-class face ID on LFW persons
# Everything not in training set = intruder (handled later at inference)
# """

# import os
# import random
# from collections import Counter

# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# from torch.utils.data import DataLoader, random_split, Subset
# from torchvision import datasets, transforms

# # -------------------
# # Config
# # -------------------
# DATA_ROOT = r"C:\Users\aldwa\OneDrive\Documents\achool\data\lfw-deepfunneled"  # <- change if needed

# IMG_SIZE = 128
# BATCH_SIZE = 64
# VAL_SPLIT = 0.2
# EPOCHS = 15
# LR = 1e-3
# WEIGHT_DECAY = 1e-4
# MIN_IMAGES_PER_CLASS = 5  # filter out identities with too few images

# SEED = 42

# # -------------------
# # Reproducibility
# # -------------------
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # -------------------
# # Check data root
# # -------------------
# if not os.path.isdir(DATA_ROOT):
#     print("ERROR: DATA_ROOT does not exist:", DATA_ROOT)
#     raise SystemExit(1)

# # -------------------
# # Transforms
# # -------------------
# base_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # -------------------
# # Load raw LFW as ImageFolder
# # -------------------
# full_dataset = datasets.ImageFolder(
#     root=DATA_ROOT,
#     transform=base_transforms
# )

# print("Total images (raw):", len(full_dataset))
# print("Total classes (raw):", len(full_dataset.classes))

# # -------------------
# # Filter classes with too few images
# # -------------------
# all_labels = [y for _, y in full_dataset]
# label_counts = Counter(all_labels)

# # keep labels with >= MIN_IMAGES_PER_CLASS
# keep_labels = {lbl for lbl, c in label_counts.items() if c >= MIN_IMAGES_PER_CLASS}
# print(f"Keeping classes with >= {MIN_IMAGES_PER_CLASS} images:", len(keep_labels), "classes")

# # map old label -> new compact label
# old_to_new = {old: i for i, old in enumerate(sorted(keep_labels))}
# new_classes = [full_dataset.classes[old] for old in sorted(keep_labels)]
# num_classes = len(new_classes)
# print("Num classes (filtered):", num_classes)

# # indices of samples belonging to kept classes
# kept_indices = [i for i, y in enumerate(all_labels) if y in keep_labels]

# print("Images after filtering:", len(kept_indices))

# class FilteredLFW(torch.utils.data.Dataset):
#     def __init__(self, base_ds, indices, old_to_new):
#         self.base_ds = base_ds
#         self.indices = indices
#         self.old_to_new = old_to_new

#         # build class list
#         all_old = sorted(old_to_new.keys())
#         self.classes = [base_ds.classes[o] for o in all_old]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         real_idx = self.indices[idx]
#         img, old_label = self.base_ds[real_idx]
#         new_label = self.old_to_new[old_label]
#         return img, new_label

# dataset = FilteredLFW(full_dataset, kept_indices, old_to_new)

# print("Final dataset size:", len(dataset))
# print("Example classes:", dataset.classes[:10])

# # -------------------
# # Quick class distribution print (first 20)
# # -------------------
# all_labels_filtered = [lbl for _, lbl in dataset]
# label_counts_filtered = Counter(all_labels_filtered)
# idx_to_class = {i: c for i, c in enumerate(dataset.classes)}

# print("Number of classes (filtered):", len(idx_to_class))
# for idx, count in list(label_counts_filtered.items())[:20]:
#     print(f"{idx_to_class[idx]}: {count}")

# # -------------------
# # Train/Val split
# # -------------------
# num_total = len(dataset)
# val_size = int(VAL_SPLIT * num_total)
# train_size = num_total - val_size

# train_dataset, val_dataset = random_split(
#     dataset,
#     [train_size, val_size],
#     generator=torch.Generator().manual_seed(SEED)
# )

# print("Train size:", len(train_dataset))
# print("Val size:", len(val_dataset))

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0,
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0,
# )

# # -------------------
# # Simple CNN
# # -------------------
# class SimpleFaceCNN(nn.Module):
#     def __init__(self, num_classes, img_size=IMG_SIZE):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 128 -> 64

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 64 -> 32

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 32 -> 16
#         )

#         feat_size = img_size // 8
#         flattened_dim = 128 * feat_size * feat_size

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(flattened_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# model = SimpleFaceCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
# print(model)

# # -------------------
# # Loss, optimizer
# # -------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# # -------------------
# # Training / evaluation helpers
# # -------------------
# def accuracy_from_logits(logits, targets):
#     preds = logits.argmax(dim=1)
#     correct = (preds == targets).float().sum()
#     return (correct / targets.size(0)).item()

# def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
#     model.train()
#     running_loss = 0.0
#     running_acc = 0.0
#     n_samples = 0

#     for batch_idx, (images, labels) in enumerate(loader):
#         images = images.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)

#         optimizer.zero_grad()
#         logits = model(images)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()

#         batch_size = labels.size(0)
#         acc = accuracy_from_logits(logits, labels)

#         running_loss += loss.item() * batch_size
#         running_acc += acc * batch_size
#         n_samples += batch_size

#         if (batch_idx + 1) % 50 == 0:
#             print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1}/{len(loader)}] "
#                   f"Loss: {loss.item():.4f} | Acc: {acc:.4f}")

#     epoch_loss = running_loss / n_samples
#     epoch_acc = running_acc / n_samples
#     return epoch_loss, epoch_acc

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     running_acc = 0.0
#     n_samples = 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             logits = model(images)
#             loss = criterion(logits, labels)

#             batch_size = labels.size(0)
#             running_loss += loss.item() * batch_size
#             running_acc += accuracy_from_logits(logits, labels) * batch_size
#             n_samples += batch_size

#     epoch_loss = running_loss / n_samples
#     epoch_acc = running_acc / n_samples
#     return epoch_loss, epoch_acc

# # -------------------
# # Training loop
# # -------------------
# best_val_acc = 0.0
# best_state = None

# print("\n=== Training CNN on LFW (filtered) ===")
# for epoch in range(1, EPOCHS + 1):
#     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#     print(f"Epoch {epoch:02d} | "
#           f"Train L: {train_loss:.4f} A: {train_acc:.4f} | "
#           f"Val L: {val_loss:.4f} A: {val_acc:.4f}")

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         best_state = model.state_dict()

# if best_state is not None:
#     model.load_state_dict(best_state)
#     torch.save(model.state_dict(), "lfw_cnn_best.pth")
#     print(f"\nLoaded best model (val acc={best_val_acc:.4f}) and saved to lfw_cnn_best.pth")
# else:
#     print("\nNo improvement over initial model; nothing saved.")



#================================================================================

"""
project_draft.py
Run in VS Code with a Python interpreter (preferably with CUDA GPU, e.g. RTX 3090).

Requirements:
    pip install torch torchvision pillow
"""

import os
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def main():
    # -------------------
    # Config
    # -------------------
    DATA_ROOT = r"C:\Users\aldwa\OneDrive\Documents\achool\data\lfw-deepfunneled"  # <- change if needed

    IMG_SIZE = 128
    BATCH_SIZE = 64
    VAL_SPLIT = 0.2
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MIN_IMAGES_PER_CLASS = 5  # filter out identities with too few images

    SEED = 42

    # -------------------
    # Reproducibility
    # -------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -------------------
    # Check data root
    # -------------------
    if not os.path.isdir(DATA_ROOT):
        print("ERROR: DATA_ROOT does not exist:", DATA_ROOT)
        raise SystemExit(1)

    # -------------------
    # Transforms
    # -------------------
    base_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -------------------
    # Load raw LFW as ImageFolder
    # -------------------
    full_dataset = datasets.ImageFolder(
        root=DATA_ROOT,
        transform=base_transforms
    )

    print("Total images (raw):", len(full_dataset))
    print("Total classes (raw):", len(full_dataset.classes))

    # -------------------
    # Filter classes with too few images
    # -------------------
    all_labels = [y for _, y in full_dataset]
    label_counts = Counter(all_labels)

    keep_labels = {lbl for lbl, c in label_counts.items() if c >= MIN_IMAGES_PER_CLASS}
    print(f"Keeping classes with >= {MIN_IMAGES_PER_CLASS} images:", len(keep_labels), "classes")

    old_to_new = {old: i for i, old in enumerate(sorted(keep_labels))}
    new_classes = [full_dataset.classes[old] for old in sorted(keep_labels)]
    num_classes = len(new_classes)
    print("Num classes (filtered):", num_classes)

    kept_indices = [i for i, y in enumerate(all_labels) if y in keep_labels]
    print("Images after filtering:", len(kept_indices))

    class FilteredLFW(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices, old_to_new):
            self.base_ds = base_ds
            self.indices = indices
            self.old_to_new = old_to_new

            all_old = sorted(old_to_new.keys())
            self.classes = [base_ds.classes[o] for o in all_old]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, old_label = self.base_ds[real_idx]
            new_label = self.old_to_new[old_label]
            return img, new_label

    dataset = FilteredLFW(full_dataset, kept_indices, old_to_new)

    print("Final dataset size:", len(dataset))
    print("Example classes:", dataset.classes[:10])

    # -------------------
    # Class distribution (first 20)
    # -------------------
    all_labels_filtered = [lbl for _, lbl in dataset]
    label_counts_filtered = Counter(all_labels_filtered)
    idx_to_class = {i: c for i, c in enumerate(dataset.classes)}

    print("Number of classes (filtered):", len(idx_to_class))
    for idx, count in list(label_counts_filtered.items())[:20]:
        print(f"{idx_to_class[idx]}: {count}")

    # -------------------
    # Train/Val split
    # -------------------
    num_total = len(dataset)
    val_size = int(VAL_SPLIT * num_total)
    train_size = num_total - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # keep 0 to avoid multiprocessing issues
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------
    # Simple CNN
    # -------------------
    class SimpleFaceCNN(nn.Module):
        def __init__(self, num_classes, img_size=IMG_SIZE):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 128 -> 64

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64 -> 32

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32 -> 16
            )

            feat_size = img_size // 8
            flattened_dim = 128 * feat_size * feat_size

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleFaceCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
    print(model)

    # -------------------
    # Loss, optimizer
    # -------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -------------------
    # Training / evaluation helpers
    # -------------------
    def accuracy_from_logits(logits, targets):
        preds = logits.argmax(dim=1)
        correct = (preds == targets).float().sum()
        return (correct / targets.size(0)).item()

    def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_samples = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            acc = accuracy_from_logits(logits, labels)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            n_samples += batch_size

            if (batch_idx + 1) % 50 == 0:
                print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1}/{len(loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.4f}")

        epoch_loss = running_loss / n_samples
        epoch_acc = running_acc / n_samples
        return epoch_loss, epoch_acc

    def evaluate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        n_samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_acc += accuracy_from_logits(logits, labels) * batch_size
                n_samples += batch_size

        epoch_loss = running_loss / n_samples
        epoch_acc = running_acc / n_samples
        return epoch_loss, epoch_acc

    # -------------------
    # Training loop
    # -------------------
    best_val_acc = 0.0
    best_state = None

    print("\n=== Training CNN on LFW (filtered) ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | "
              f"Train L: {train_loss:.4f} A: {train_acc:.4f} | "
              f"Val L: {val_loss:.4f} A: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "lfw_cnn_best.pth")
        print(f"\nLoaded best model (val acc={best_val_acc:.4f}) and saved to lfw_cnn_best.pth")
    else:
        print("\nNo improvement over initial model; nothing saved.")


if __name__ == "__main__":
    main()
