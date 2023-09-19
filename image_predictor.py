import os
import glob
import heapq
import time
from PIL import Image
import torch
import torchvision.models
import torchvision.transforms as transforms

class ImagePredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.model = torchvision.models.resnet50()
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    def prepare_image(self, image):
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, image):
        image = self.prepare_image(image)
        with torch.no_grad():
            preds = self.model(image)
        return preds.item()


def main():
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = "images"
    model_path = 'model-resnet50.pth'
    predictor = ImagePredictor(model_path, device)

    scores = []
    for image_file in glob.iglob(os.path.join(image_dir, '*.*')):
        image = Image.open(image_file)
        score = predictor.predict(image)
        scores.append((-score, image_file))

    heapq.heapify(scores)

    while scores:
        score, image_file = heapq.heappop(scores)
        print(f"Popularity Score: {-score}  {image_file}")

    end_time = time.time()
    print(f"Program executed in: {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
