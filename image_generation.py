from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")  # Chạy trên GPU

def generate_image(prompt):
    """Tạo ảnh từ mô tả văn bản."""
    image = pipe(prompt).images[0]
    image.show()

generate_image("Một thành phố tương lai với ánh đèn neon")
