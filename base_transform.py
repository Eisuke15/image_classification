from torchvision import transforms


class BaseTransform():

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),  # the length of the shorter side becomes `resize`
            transforms.CenterCrop(resize),  # crop the center of the image by resize x resize
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    def __call__(self, img):
        return self.base_transform(img)