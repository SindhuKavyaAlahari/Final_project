import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torchvision
import torch
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")

        # Parse the annotation file
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.retinanet.RetinaNetHead(in_features, num_classes)
    return model

def train_model(model, dataloader, num_epochs=10):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {losses.item()}")

    return model

# Load your dataset
dataset = CustomDataset(root='/Users/sindhukavyaalahari/Documents/herts/finalproject/sindhu/coco')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize and train the model
num_classes = len(dataset.classes)  # Number of classes in your dataset
model = get_model(num_classes)
trained_model = train_model(model, dataloader)

# Save the trained model
torch.save(trained_model.state_dict(), 'model.pth')
