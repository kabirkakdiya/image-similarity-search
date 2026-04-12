import io, torch
from PIL import Image
from torchvision import transforms

class DinoEmbedder:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            print("Falling back to CPU")

        self.preprocess = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("Loading DINOv2 model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("DinoEmbedder ready")

    @torch.no_grad()
    def embed(self, image_input: str | bytes | Image.Image) -> list[float]:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError("image_input must be str path, bytes, or PIL Image")

        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        embedding = self.model(input_tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.cpu().numpy()[0].tolist()