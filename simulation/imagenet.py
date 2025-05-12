import os 
import sys
import argparse
import torch
import torchvision
from torchvision import transforms, datasets

from tqdm import tqdm
import json


PATH = os.getenv("DQPATH", '')
sys.path.append(PATH)
sys.path.append(os.path.join(PATH, "backend"))
from backend import convert_model, QuantMode, CorruptTransform


from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CachedImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        # Open the image file within a context manager to ensure it's closed immediately
        with open(path, 'rb') as f:
            img = Image.open(f)
            image = img.convert('RGB')
            # Ensure image data is loaded before closing the file
            img.load()
        # Apply transforms after the file is closed
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ResNet50 on ImageNet')
    parser.add_argument('--data', metavar='DIR', required=True,
                        help='Path to ImageNet validation folder (expects subdirectories for each class)')
    parser.add_argument('--b', default=256, type=int,
                        help='Mini-batch size for evaluation (default: 256)')
    parser.add_argument('--wk', default=8, type=int,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to use (default: None, uses CPU)')
    parser.add_argument('--model', default='resnet50', help='Available models: resnet50, mobilenetv3')
    args = parser.parse_args()

    mode = eval(os.getenv("dq", "0"))
    corrupt = eval(os.getenv("corrupt", "0")) == 1
    per_channel = eval(os.getenv("ch", "1")) == 1

    sampling_stride = eval(os.getenv("s_s", "1")) 
    c_std = eval(os.getenv("c_std", "3"))
    l_std = eval(os.getenv("l_std", "3"))

    cal_size = eval(os.getenv("cal_size", "16"))
    verb = eval(os.getenv("verb", "0"))
    
    seed = eval(os.getenv("seed", "42"))


    print()
    print("LOG+++++ ESTIMATE MODE =", mode)
    print("LOG+++++ Corrupt =", corrupt)
    print("LOG+++++ Per_Channel =", per_channel)

    confss = {
        'global':{
            'def': {'per_channel':per_channel},
            QuantMode.ESTIMATE: {
                'Conv2d': {
                    'e_std': c_std,
                    'sampling_stride': sampling_stride,
                    'cal_size': cal_size
                },
                'Linear': {
                    'e_std': l_std,
                    'cal_size': cal_size
                }
            },
            QuantMode.STATIC: {'cal_size': cal_size}

        },
        'layers':{
            'classifier': {'skip': True},
        }
    }
    
    # Set device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        else:
            print("CUDA is not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    if corrupt:
        pass
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            CorruptTransform(None, (3,5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    # Load the validation dataset
    val_dataset = datasets.ImageFolder(root=f"{args.data}/val", transform=preprocess)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.wk,
        pin_memory=True
    )

    # Define where to save the cached subset. Here we use a JSON file.
    cache_file = os.path.join(args.data, f'subset_cache_{seed}_{cal_size}.json')

    if os.path.exists(cache_file):
        # Load the cached list of (filepath, label) pairs.
        print(f'LOADED subset_cache_{seed}_{cal_size}.json')
        with open(cache_file, 'r') as f:
            cached_samples = json.load(f)
    else:
        # First run: Use ImageFolder to gather all samples.
        full_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'train'))
        generator = torch.Generator().manual_seed(seed)
        # Generate a random permutation of indices and select cal_size
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()[:cal_size]
        
        print(f'GENERATED {len(indices)} samples with seed {seed}')
        
        # Extract only the samples for the selected indices.
        # (Each sample is a tuple: (path, label))
        cached_samples = [full_dataset.samples[i] for i in indices]
        
        # Save the cached samples to a JSON file.
        with open(cache_file, 'w') as f:
            json.dump(cached_samples, f)

    # Create the subset dataset using the cached sample list.
    cal_subset = CachedImageDataset(cached_samples, transform=preprocess)

    cal_loader = DataLoader(
        cal_subset,
        batch_size=8,
        shuffle=False,
        num_workers=args.wk,
        pin_memory=True,
        # Optional: Reduce memory usage with lower prefetch factor
        prefetch_factor=2 if args.wk > 0 else None,
        persistent_workers=True if args.wk > 0 else False
    )
    
    # Load the pretrained ResNet50 model and set to evaluation mode
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif args.model == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise RuntimeError('incorrect model')

    model.to(device)
    model.eval()
    
    if mode == 1:
        model = convert_model(
            model,
            mode=QuantMode.ESTIMATE,
            config=confss
        )

    elif mode == 2:
        model = convert_model(
            model,
            mode=QuantMode.DYNAMIC,
            config=confss
        )
    
    elif mode == 3:
        model = convert_model(
            model,
            mode=QuantMode.STATIC,
            config=confss
        )

    if mode == 1 or mode == 3:
        for (images, _) in tqdm(cal_loader, desc="CALIBRATING", total=len(cal_loader)): 
            images = images.to(device)          
            output = model(images)

    print(model)
    

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for (images, targets) in tqdm(val_loader, desc='Testing', total=len(val_loader)):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            # Get top-5 predictions
            _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            total += targets.size(0)
            # Top-1: compare first prediction to ground-truth
            top1_correct += (pred[:, 0] == targets).sum().item()
            # Top-5: check if target is within the top-5 predictions
            
            top5_correct += (pred.eq(targets.view(-1, 1))).sum().item()

    acc_1 = 100.0 * top1_correct / total
    acc_5 = 100.0 * top5_correct / total

    print()
    print("Top 1 Acc: ", acc_1)
    print("Top 5 Acc: ", acc_5)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    results_log = f"results/{seed}/{str(QuantMode(mode)).split('.')[1]}__corr_{corrupt}_per_channel_{per_channel}"
    if mode==0:
        results_log = f"{results_log.split('_per_channel_')[0]}"
    os.makedirs(results_log, exist_ok=True)

    results_file = f"{args.model}_.json"
    if mode==1: #se estimate aggiungo il sampling stride al nome del file
        results_file = f"{results_file.split('.json')[0]}_stride_{sampling_stride}_c{c_std}_l{l_std}.json"
    if mode==3 or mode==1: #se static aggiungo il cal_size al nome del file
        results_file = f"{results_file.split('.json')[0]}_cal_{cal_size}.json"
        
    results = [{"top1": acc_1, "top5": acc_5}]

    with open(os.path.join(results_log, results_file), "w") as fout:
        json.dump(results, fout)

    print()
    print(f"++++++ Saved results in {os.path.join(results_log, results_file)}")