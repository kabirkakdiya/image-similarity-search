# OPTIMIZED APPROACH
import io, torch, psutil
from PIL import Image
from torchvision import transforms

class DinoEmbedder:
    def __init__(self):
        # 1. Hardware Detection & Thread Tuning
        self.physical_cores = psutil.cpu_count(logical=False)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16
            print(f"Using MPS (GPU)")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            # Maximize CPU usage if no GPU is found
            torch.set_num_threads(self.physical_cores)
            print(f"Falling back to CPU using {self.physical_cores} threads")

        # 2. Faster Preprocessing
        # Using Bilinear and Avoiding redundant conversions
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("Loading DINOv2 ViT-L/14...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # 3. Memory Optimization for Mac
        if self.device.type == "mps":
            # Warmup is essential to "cache" the Metal shaders
            self._warmup()

    @torch.no_grad()
    def _warmup(self):
        dummy = torch.randn(1, 3, 224, 224).to(self.device, dtype=self.dtype)
        for _ in range(3): self.model(dummy)

    def embed(self, image_input: str | bytes | Image.Image) -> list[float]:
        if isinstance(image_input, (str, bytes)):
            img = Image.open(io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input).convert('RGB')
        else:
            img = image_input

        # Move tensor to device and cast to correct dtype immediately
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        
        embedding = self.model(input_tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        
        # Convert back to float32 for the list return to maintain precision in JSON
        return embedding.detach().cpu().to(torch.float32).numpy()[0].tolist()


# import io, torch
# from PIL import Image
# from torchvision import transforms

# class DinoEmbedder:
#     def __init__(self):
#         if torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#             print("Using MPS")
#         elif torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             print("Using CUDA")
#         else:
#             self.device = torch.device("cpu")
#             print("Falling back to CPU")

#         self.preprocess = transforms.Compose([
#             transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         print("Loading DINOv2 model...")
#         self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         print("DinoEmbedder ready")

#     @torch.no_grad()
#     def embed(self, image_input: str | bytes | Image.Image) -> list[float]:
#         if isinstance(image_input, str):
#             img = Image.open(image_input).convert('RGB')
#         elif isinstance(image_input, bytes):
#             img = Image.open(io.BytesIO(image_input)).convert('RGB')
#         elif isinstance(image_input, Image.Image):
#             img = image_input.convert('RGB')
#         else:
#             raise ValueError("image_input must be str path, bytes, or PIL Image")

#         input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
#         embedding = self.model(input_tensor)
#         embedding = torch.nn.functional.normalize(embedding, dim=1)
#         return embedding.cpu().numpy()[0].tolist()
