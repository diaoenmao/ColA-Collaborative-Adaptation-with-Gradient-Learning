import numpy as np
import os
import pickle
import torch
from config import cfg
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class DreamBoothDataset(Dataset):
    data_name = 'DreamBooth'
    file = [('https://drive.google.com/file/d/1jPpJcGUH68o7LIMM9asS0cs4Cec4NqYz/view?usp=drive_link', None)]
    
    def __init__(
            self, 
            root, 
            split, 
            model,
            tokenizer,
            instance_data_dir,
            instance_prompt,
            class_data_dir=None,
            class_prompt=None,
            transform=None
        ):
        self.root = os.path.expanduser(root)
        self.split = split
        self.tokenizer = tokenizer
        self.instance_data_dir = os.path.join(self.processed_folder, instance_data_dir)
        self.instance_prompt = instance_prompt
        self.class_data_dir = os.path.join(self.processed_folder, class_data_dir)
        self.class_prompt = class_prompt
        self.transform = transform

        if not check_exists(self.processed_folder):
            self.download()
        if not check_exists(self.class_data_dir):
            self.process(model)
        self.make_data()
        # self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
        #                                        mode='pickle')
        # self.other = {}
        # self.classes_counts = make_classes_counts(self.target)
        # self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def __getitem__(self, index):

        input = {}

        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        input["instance_images"] = self.transform(instance_image)
        input["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            input["class_images"] = self.transform(class_image)
            input["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return input

    def __len__(self):
        return self._length
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def generate_prior_data(self, model):
        model_name = cfg['model_name']
        model.to(cfg['device'])
        for i in range(cfg[cfg['model_name']]['num_class_image']):
            prompt = self.class_prompt
            image = model(prompt, num_inference_steps=cfg[model_name]['num_inference_steps'], guidance_scale=cfg[model_name]['guidance_scale']).images[0]
            image_path = os.path.join(self.class_data_dir, f"class_pic_{i}.png")
            # Save the image to the specified path
            image.save(image_path)
        return

    def process(self, model):
        makedir_exist_ok(self.class_data_dir)
        self.generate_prior_data(model)
        # train_set, test_set, meta = self.make_data()
        # save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        # save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        # save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, os.path.join(self.raw_folder, filename), md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        a = Path(self.instance_data_dir).iterdir()
        b = list(Path(self.instance_data_dir).iterdir())
        self.instance_images_path = list(Path(self.instance_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if self.class_data_dir is not None:
            self.class_images_path = list(Path(self.class_data_dir).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        return None
