import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_loaders(config, inference=True):
    img_size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]
    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.Imagenette(root=train_dir, split='train', size='full', download=True, transform=train_transform)
    val_dataset = datasets.Imagenette(root=val_dir, split='val', size='full', download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

