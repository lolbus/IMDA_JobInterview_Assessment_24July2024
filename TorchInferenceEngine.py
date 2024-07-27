import PIL
from PIL import Image
from SimpleCNN import SimpleCNNArchitecture, torch_transform
import torch
# Define the coordinates for each character in the captcha images
coordinates = [
    (4, 10, 14, 22),  # Coordinates for the first character
    (13, 10, 23, 22),  # Coordinates for the second character
    (22, 10, 32, 22),  # Coordinates for the third character
    (31, 10, 41, 22),  # Coordinates for the fourth character
    (40, 10, 50, 22)   # Coordinates for the fifth character
]
        
# Generate Inference data from a random captcha image
def crop_image_and_return_5char(image, coordinates:list)->list:
    """Converts an captcha image into 5 character images for analysis character by character,
    args:
    input: captcha image shape of 30 by 60, static coordinate list consistent throughout the nature of the problem
    output: list of 5 images each shape of 10 by 12"""
    # Define the coordinates for each character in the captcha images
    image_list = []
    # Trim each character and save them with the desired filenames
    for i, (x1, y1, x2, y2) in enumerate(coordinates):
        cropped_image = image.crop((x1, y1, x2, y2))
        image_list.append(cropped_image)
    return image_list

def tensor_to_char(tensor):
    """Maps predicted tensor into predicted label character
    args:
    input: label/predicted tensor
    output: A character string:str"""
    index = tensor.item()  # Get the integer value from the tensor
    if 0 <= index <= 25:
        return chr(index + ord('A'))  # Convert to character 'A' to 'Z'
    elif 26 <= index <= 35:
        return chr(index - 26 + ord('0'))  # Convert to character '0' to '9'
    else:
        raise ValueError("Index out of range for character mapping")

class load_torch_ai_model(object):
    def __init__(self, model_path:str):
        torch_model = SimpleCNNArchitecture()
        torch_model.load_state_dict(torch.load(model_path))
        torch_model.eval()
        self.torch_model = torch_model

    def get_image_information(self, image_path)->dict:
        """
        Main inference caller
        args: 
            input_path: str
        output: 
            dict containing inference result:dict
        """
        input_image = Image.open(image_path)
        image_char_list = crop_image_and_return_5char(input_image, coordinates)
        inference_result = {'InferredCharacters':""}
        for image_char in image_char_list:
            transformed_image_char = torch_transform(image_char.convert('L'))
            with torch.no_grad():
                output = self.torch_model(transformed_image_char)
                _, predicted = torch.max(output, 1)
                predicted_char = tensor_to_char(predicted)
                inference_result['InferredCharacters'] += predicted_char
        return inference_result
