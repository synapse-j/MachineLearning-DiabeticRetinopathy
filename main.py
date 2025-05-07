import pandas as pd
import os
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# --- Load CSV and create label dictionary ---
df = pd.read_csv('trainLabels_cropped.csv')  # Assumes columns: Unnamed:0, image, level
labels_dict = dict(zip(df['image'] + '.jpeg', df['level']))

# --- Dataset class for paired images ---
class PairedEyeDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_images=None, label_file=None):
        self.folder_path = folder_path
        self.transform = transform
        all_files = os.listdir(folder_path)

        left_images = sorted([f for f in all_files if f.endswith('left.jpeg')])
        right_images = sorted([f for f in all_files if f.endswith('right.jpeg')])

        self.pairs = []
        for left in left_images:
            id_part = left.replace('_left.jpeg', '')
            right = id_part + '_right.jpeg'
            if right in right_images:
                self.pairs.append((left, right))

        if max_images:
            self.pairs = self.pairs[:max_images]

        self.labels = label_file if label_file else {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left_file, right_file = self.pairs[idx]
        left_path = os.path.join(self.folder_path, left_file)
        right_path = os.path.join(self.folder_path, right_file)

        left_image = Image.open(left_path).convert('RGB')
        right_image = Image.open(right_path).convert('RGB')

        left_label = self.labels.get(left_file, 0)
        right_label = self.labels.get(right_file, 0)
        label = max(left_label, right_label)  # Use the more severe DR level

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        combined = torch.cat([left_image, right_image], dim=0)
        return combined, label

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Dataset and loader ---
dataset = PairedEyeDataset('resized_train_cropped', transform=transform, max_images=200, label_file=labels_dict)
loader = DataLoader(dataset, batch_size=4, num_workers=0)

# --- Class weights ---
label_list = [label for _, label in dataset]
label_counts = Counter(label_list)
num_classes = 5
class_counts = [label_counts.get(i, 1) for i in range(num_classes)]
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float)

# --- Model setup ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train():
    for epoch in range(5):  # Train for more epochs
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), 'dr_model_paired.pth')

def evaluate():
    model.eval()
    all_images = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_images.extend(images.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Label distribution in predictions:")
    print("Actual labels distribution:", Counter(all_labels))
    print("Predicted labels distribution:", Counter(all_preds))

    stage_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe NPDR (3)', 'Proliferative (4)']

    selected = {i: None for i in range(len(stage_names))}
    for idx, (img, label, pred) in enumerate(zip(all_images, all_labels, all_preds)):
        if label == pred and selected[label] is None:
            filename = f"{dataset.pairs[idx][0]} & {dataset.pairs[idx][1]}"
            selected[label] = (img, label, pred, filename)

    # --- Create figure with better layout ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 1, 1.1])  # 3 rows, 5 columns

    # --- Row 0: Example Images (1 per class) ---
    for i in range(5):
        if selected[i] is None:
            continue
        img, true_label, pred_label, fname = selected[i]
        left = img[:3]
        right = img[3:]
        left = np.transpose(left, (1, 2, 0))
        right = np.transpose(right, (1, 2, 0))
        left = (left + 1) / 2
        right = (right + 1) / 2
        combined = np.hstack([left, right])

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(combined)
        ax.set_title(f"{fname}\nActual: {stage_names[true_label]}\nPred: {stage_names[pred_label]}", fontsize=8)
        ax.axis('off')

    # --- Row 1: Confusion Matrix (spans 3 columns) ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(stage_names))))
    ax_cm = fig.add_subplot(gs[1, :3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stage_names)
    disp.plot(ax=ax_cm, cmap='Blues', xticks_rotation=45, values_format='d')
    ax_cm.set_title("Confusion Matrix")

    # --- Row 1: Class Distribution Bar Chart (spans 2 columns) ---
    class_counts = Counter(all_labels)
    class_distribution = [class_counts.get(i, 0) for i in range(len(stage_names))]
    ax_bar = fig.add_subplot(gs[1, 3:])
    ax_bar.bar(range(len(stage_names)), class_distribution, color='skyblue')
    ax_bar.set_title("Class Distribution")
    ax_bar.set_ylabel("Sample Count")
    ax_bar.set_xticks(range(len(stage_names)))
    ax_bar.set_xticklabels(stage_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


    # --- Create a new figure for Detection Rate & False Positive Rate ---
    fig2 = plt.figure(figsize=(10, 6))
    ax_rates = fig2.add_subplot(111)  # Single axes

    # Calculate Detection Rate and False Positive Rate
    detection_rate = []
    false_positive_rate = []
    for i in range(len(stage_names)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        detection_rate.append(tp / (tp + fn) if tp + fn > 0 else 0)
        false_positive_rate.append(fp / (fp + tn) if fp + tn > 0 else 0)

    # Plot
    width = 0.35
    x = np.arange(len(stage_names))
    ax_rates.bar(x - width/2, detection_rate, width, label='Detection Rate', color='g')
    ax_rates.bar(x + width/2, false_positive_rate, width, label='False Positive Rate', color='blue')

    ax_rates.set_title("Detection Rate & False Positive Rate")
    ax_rates.set_xlabel("Classes")
    ax_rates.set_ylabel("Rates")
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(stage_names, rotation=45, ha='right')
    ax_rates.legend()

    plt.tight_layout()
    plt.show()



# Main execution
if __name__ == '__main__':
    print(f"Number of paired samples: {len(dataset)}")
    train()
    evaluate()
