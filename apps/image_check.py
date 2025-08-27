from PIL import Image

# Load your image
img = Image.open("imgs/demo.png")  
# Get size
width, height = img.size

print(f"Image size: {width} x {height} pixels")
