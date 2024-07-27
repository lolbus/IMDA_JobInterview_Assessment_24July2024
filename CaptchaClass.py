from PIL import Image
import torch
import numpy as np

"""
Available Strategies:
1. OpenAI GPT4o AI Assistant - "LLMAIAssistant"
2. PyTorch model self trained model - "PytorchCNN"
3. Pytorch model + OpenAI GPT4o AI Assistant Collaboration (Final Solution) - "Collaborative"
4. Pytorch model trained as O, 0, 1, I 4-class Discriminator + OpenAI GPT4o AI Assistant Collaboration - "Collaborative-with-NicheDiscriminator"
"""

def convert_image_to_white_or_black(im_path):
    """
    This function convert an image file to a black (0 value) or white (255 value) image and save it as a png
    args:
        img : im_path
    output:
        result_im_path: converted image im_path
    """
    image = Image.open(im_path)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    binary_tensor = (image_tensor >= 128).to(torch.uint8)
    result_tensor = binary_tensor * 255
    result_np = result_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    result_image = Image.fromarray(result_np)
    sp = r".\last_scan_white_or_black_image.png"
    result_image.save(sp, format='PNG')
    return sp

class Captcha(object):
    def __init__(self, strategy = "Collaborative", openai_api_key = ""):
        self.white_or_black = False
        if strategy == "LLMAIAssistant":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            # Use data used for zero-shot prompting to fine-tune LLM
            finetune_input_dir = r"C:\Users\guang\OneDrive\Desktop\IMDA\fine-tune-data\input"
            finetune_output_dir = r"C:\Users\guang\OneDrive\Desktop\IMDA\fine-tune-data\output"
            training_data = dataloader(input_dir = finetune_input_dir, output_dir = finetune_output_dir)
            OpenAIInferenceEngine.llm_agent = llm_model(finetune = False, training_data = training_data, openai_api_key=openai_api_key)
            self.model = load_OpenAI_ai_model()
        elif strategy == "PytorchCNN":
            from TorchInferenceEngine import load_torch_ai_model
            self.model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver2.pth')
        elif strategy == "PytorchCNN-WORB":
            from TorchInferenceEngine import load_torch_ai_model
            self.model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver4.pth')
            self.white_or_black = True
        elif strategy == "Collaborative":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            from TorchInferenceEngine import load_torch_ai_model
            from CollaborativeInferenceEngine import load_collaborative_model
            OpenAIInferenceEngine.llm_agent = llm_model(openai_api_key=openai_api_key)
            openai_ai_model = load_OpenAI_ai_model()
            torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver2.pth')   
            self.model = load_collaborative_model(torch_ai_model=torch_ai_model, openai_ai_model=openai_ai_model)
        elif strategy == "Collaborative-WORB":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            from TorchInferenceEngine import load_torch_ai_model
            from CollaborativeInferenceEngine import load_collaborative_model
            OpenAIInferenceEngine.llm_agent = llm_model(openai_api_key=openai_api_key)
            openai_ai_model = load_OpenAI_ai_model()
            torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver4.pth')   
            self.model = load_collaborative_model(torch_ai_model=torch_ai_model, openai_ai_model=openai_ai_model)
            self.white_or_black = True
        elif strategy == "Collaborative-with-NicheDiscriminator":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            from TorchInferenceEngine import load_torch_ai_model
            from CollaborativeInferenceEngine import load_collaborative_model
            OpenAIInferenceEngine.llm_agent = llm_model(openai_api_key=openai_api_key)
            openai_ai_model = load_OpenAI_ai_model()
            torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_O01IDiscriminator_model_ver3.pth')   
            self.model = load_collaborative_model(torch_ai_model=torch_ai_model, openai_ai_model=openai_ai_model)
        else:
            print(f"{strategy} is not a valid method")
        self.caller = self.model.get_image_information

    def __call__(self, im_path, save_path = None):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        if self.white_or_black:
            im_path = convert_image_to_white_or_black(im_path)
        result = self.caller(im_path)
        if save_path != None:
            with open(save_path, 'w') as sf:
                print(f"writing to {save_path}")
                sf.write(result['InferredCharacters'])
        return result