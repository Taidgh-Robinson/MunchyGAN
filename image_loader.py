from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import torch 

def load_all_images(directory="C:\\Users\\taidg\\python\\ML\\MunchGAN\\data\\images"):
    files = os.listdir(directory)
    images = [] 
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            image = Image.open(filepath)
            images.append(image)
        except Exception as e:
            print(f"Error loading image '{filename}': {e}")

    return images

def get_max_width_and_height_and_area(images): 
    h = 0 
    w = 0 
    area = 0 

    for _, img in enumerate(images):
        width, height = img.size
        if width > w: 
            w = width
        if height > h: 
            h = height
        if width * height > area: 
            area = width * height

    return (w, h, area)

def preprocess_iamge(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize the image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize using mean and std
    ])

    return preprocess(image)




def inverse_transform(preprocessed_image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inverse_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
    )

    denormalized_image = inverse_normalize(preprocessed_image)
    pil_image = transforms.ToPILImage()(denormalized_image).convert("RGB")    
    return pil_image


def save_images_as_np(images):
    for idx, image in enumerate(images):
        preproccessed_image = preprocess_iamge(image)
        numpy_array = preproccessed_image.numpy()
        np.save("data/preprocessed_datas/" + str(idx) + ".npy", numpy_array)

image = np.load('data/preprocessed_data/17.npy')
tensor = torch.tensor(image)
img = inverse_transform(tensor)
img.show()

