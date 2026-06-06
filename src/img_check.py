# 이미지 확인하기
import matplotlib.pyplot as plt
import random
from torchvision import datasets
import yaml

with open("configs/resnet50_imagenette.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


test_dataset = datasets.Imagenette(root=config["data"]["train_dir"], split='train', size='full', download=True)
random_idx = random.randint(0, len(test_dataset)-1)
image, label = test_dataset[random_idx]

print(f"{random_idx} 번째 이미지 크기 (가로,세로) : {image.size}")
plt.imshow(image)
plt.title(f"Index: {random_idx} | Label: {test_dataset.classes[label][0]}")
plt.axis('off')
plt.show()