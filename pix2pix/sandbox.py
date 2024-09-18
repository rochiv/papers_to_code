import os


root_dir_path = os.getcwd()

train_dir_path = os.path.join(root_dir_path, "cityscapes/train")    
val_dir_path = os.path.join(root_dir_path, "cityscapes/val")

print(f"Root directory: {root_dir_path}")
print(f"Train directory: {train_dir_path}")
print(f"Validation directory: {val_dir_path}")

print(f"Number of files in cityscapes/train: {len(os.listdir(train_dir_path))}")
print(f"Number of files in cityscapes/val: {len(os.listdir(val_dir_path))}")
