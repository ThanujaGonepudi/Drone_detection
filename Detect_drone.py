import torch
import os
input_img = r'input_images/drone_image.jpg'
model = torch.hub.load(os.getcwd(), 'custom', source ='local', path=r'weights/Drone_weights.pt', force_reload = True)
result = model(input_img)
result.show()