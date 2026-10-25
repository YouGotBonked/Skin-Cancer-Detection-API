import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

DEFAULT_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class Classifier:
    def __init__(self, weights_path: str, classes: list[str]):
        absolute_path = Path(weights_path).resolve()

        if not absolute_path.exists():
            raise FileNotFoundError(f"Model weight file not found at: {absolute_path}")

        self.model = torch.jit.load(str(absolute_path), map_location="cpu").eval()
        self.classes = classes

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict[str, float]:
        x = _preprocess(img.convert("RGB")).unsqueeze(0)
        probs = torch.softmax(self.model(x), dim=1)[0].tolist()
        return dict(zip(self.classes, probs))
