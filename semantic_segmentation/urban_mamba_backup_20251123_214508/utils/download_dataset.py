"""
Dataset Downloader for UrbanMamba
Downloads and prepares urban scene datasets for semantic segmentation
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib.request

class DatasetDownloader:
    """Handle dataset downloading and extraction"""
    
    def __init__(self, root_dir='./datasets'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, dest_path, desc="Downloading"):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """Extract zip or tar archive"""
        print(f"\nExtracting {archive_path.name}...")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            print("✓ Extraction complete")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def download_cityscapes_sample(self):
        """Download Cityscapes sample dataset (smaller, good for testing)"""
        print("\n" + "="*70)
        print("  Downloading Cityscapes Sample Dataset")
        print("="*70)
        print("\nNote: Full Cityscapes requires registration.")
        print("Downloading demo/sample dataset for testing...\n")
        
        # Create cityscapes directory
        dataset_dir = self.root_dir / 'cityscapes'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample images and labels (you can replace with actual Cityscapes after registration)
        print("Creating sample urban dataset structure...")
        
        # Create directory structure
        (dataset_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
        print(f"  Location: {dataset_dir}")
        print("\nTo use full Cityscapes dataset:")
        print("1. Register at: https://www.cityscapes-dataset.com/")
        print("2. Download leftImg8bit_trainvaltest.zip (11GB)")
        print("3. Download gtFine_trainvaltest.zip (241MB)")
        print(f"4. Extract to: {dataset_dir}")
        
        return dataset_dir
    
    def download_camvid(self):
        """Download CamVid dataset (smaller, urban scenes)"""
        print("\n" + "="*70)
        print("  Downloading CamVid Dataset")
        print("="*70)
        print("\nCamVid: 701 images of urban road scenes")
        print("Classes: Road, Building, Tree, Sky, Car, etc.")
        print("Size: ~500MB\n")
        
        dataset_dir = self.root_dir / 'camvid'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # CamVid download links (from official sources)
        urls = {
            'train': 'https://github.com/alexgkendall/SegNet-Tutorial/raw/master/CamVid/train.txt',
            'val': 'https://github.com/alexgkendall/SegNet-Tutorial/raw/master/CamVid/val.txt',
            'test': 'https://github.com/alexgkendall/SegNet-Tutorial/raw/master/CamVid/test.txt',
        }
        
        print("Downloading CamVid dataset splits...")
        
        for split, url in urls.items():
            dest = dataset_dir / f'{split}.txt'
            if not dest.exists():
                print(f"  Downloading {split}.txt...")
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"  ✓ {split}.txt downloaded")
                except:
                    print(f"  ✗ Failed to download {split}.txt")
        
        # Download images (this is a placeholder - CamVid typically needs manual download)
        print("\n✓ Dataset structure prepared")
        print(f"  Location: {dataset_dir}")
        print("\nCamVid full download:")
        print("1. Visit: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/")
        print("2. Or use: https://github.com/alexgkendall/SegNet-Tutorial")
        print(f"3. Extract to: {dataset_dir}")
        
        return dataset_dir
    
    def create_sample_dataset(self):
        """Create a small sample dataset for immediate testing"""
        print("\n" + "="*70)
        print("  Creating Sample Dataset for Testing")
        print("="*70)
        
        import torch
        import numpy as np
        from PIL import Image
        
        dataset_dir = self.root_dir / 'sample_urban'
        
        # Create directories
        for split in ['train', 'val']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating synthetic urban images...")
        
        # Generate sample images
        num_train = 100
        num_val = 20
        
        def generate_urban_sample(size=(512, 512)):
            """Generate synthetic urban scene"""
            # Create synthetic image
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            # Create synthetic label (6 classes: road, building, tree, water, vehicle, other)
            label = np.random.randint(0, 6, size, dtype=np.uint8)
            
            # Add some structure (simulate road, buildings, etc.)
            # Road at bottom
            label[size[0]//2:, :] = 0  # Road
            
            # Buildings in middle
            label[size[0]//4:size[0]//2, :] = 1  # Building
            
            # Sky/trees at top
            label[:size[0]//4, :size[1]//2] = 2  # Tree
            label[:size[0]//4, size[1]//2:] = 5  # Other/sky
            
            return img, label
        
        # Generate training samples
        print(f"  Generating {num_train} training samples...")
        for i in range(num_train):
            img, label = generate_urban_sample()
            
            img_pil = Image.fromarray(img)
            label_pil = Image.fromarray(label)
            
            img_pil.save(dataset_dir / 'images' / 'train' / f'sample_{i:04d}.png')
            label_pil.save(dataset_dir / 'labels' / 'train' / f'sample_{i:04d}.png')
        
        print(f"  ✓ {num_train} training samples created")
        
        # Generate validation samples
        print(f"  Generating {num_val} validation samples...")
        for i in range(num_val):
            img, label = generate_urban_sample()
            
            img_pil = Image.fromarray(img)
            label_pil = Image.fromarray(label)
            
            img_pil.save(dataset_dir / 'images' / 'val' / f'sample_{i:04d}.png')
            label_pil.save(dataset_dir / 'labels' / 'val' / f'sample_{i:04d}.png')
        
        print(f"  ✓ {num_val} validation samples created")
        
        # Create class mapping file
        class_info = {
            0: {'name': 'road', 'color': [128, 64, 128]},
            1: {'name': 'building', 'color': [70, 70, 70]},
            2: {'name': 'tree', 'color': [107, 142, 35]},
            3: {'name': 'water', 'color': [0, 0, 142]},
            4: {'name': 'vehicle', 'color': [0, 0, 142]},
            5: {'name': 'other', 'color': [220, 20, 60]},
        }
        
        import json
        with open(dataset_dir / 'classes.json', 'w') as f:
            json.dump(class_info, f, indent=2)
        
        print("\n✓ Sample dataset created successfully!")
        print(f"  Location: {dataset_dir}")
        print(f"  Training samples: {num_train}")
        print(f"  Validation samples: {num_val}")
        print(f"  Classes: {len(class_info)}")
        
        return dataset_dir
    
    def list_datasets(self):
        """List available/downloaded datasets"""
        print("\n" + "="*70)
        print("  Available Datasets")
        print("="*70)
        
        if not self.root_dir.exists():
            print("\nNo datasets directory found.")
            return
        
        datasets = list(self.root_dir.iterdir())
        
        if not datasets:
            print("\nNo datasets downloaded yet.")
        else:
            print()
            for dataset in datasets:
                if dataset.is_dir():
                    # Count files
                    img_count = len(list(dataset.rglob('*.png'))) + len(list(dataset.rglob('*.jpg')))
                    print(f"  ✓ {dataset.name}: {img_count} images")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  UrbanMamba Dataset Downloader")
    print("="*70)
    
    downloader = DatasetDownloader(root_dir='./datasets')
    
    print("\nAvailable options:")
    print("1. Create sample dataset (for immediate testing)")
    print("2. Download Cityscapes (requires registration)")
    print("3. Download CamVid")
    print("4. List downloaded datasets")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                dataset_dir = downloader.create_sample_dataset()
                print("\n✓ Ready to train with sample dataset!")
                print(f"\nUpdate config.yaml with:")
                print(f"  data_root: {dataset_dir}")
                break
                
            elif choice == '2':
                dataset_dir = downloader.download_cityscapes_sample()
                print("\n✓ Cityscapes structure ready")
                break
                
            elif choice == '3':
                dataset_dir = downloader.download_camvid()
                print("\n✓ CamVid structure ready")
                break
                
            elif choice == '4':
                downloader.list_datasets()
                continue
                
            elif choice == '5':
                print("\nExiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break
    
    print("\n" + "="*70)
    print("  Dataset Setup Complete")
    print("="*70)
    print("\nNext steps:")
    print("1. Update config.yaml or config_gpu.yaml with dataset path")
    print("2. Run: python train.py --config config_gpu.yaml")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
