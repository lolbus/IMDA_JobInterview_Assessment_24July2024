"""Available Strategies:
1. OpenAI GPT4o AI Assistant - LLMAIAssistant
2. PyTorch self trained model - PytorchCNN
3. Pytorch + OpenAI GPT4o AI Assistant Collaboration - Collaborative"""

class Captcha(object):
    def __init__(self, strategy = "Collaborative", openai_api_key = ""):
        if strategy == "LLMAIAssistant":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            # Use data used for zero-shot prompting to fine-tune LLM
            finetune_input_dir = r"C:\Users\guang\OneDrive\Desktop\IMDA\fine-tune-data\input"
            finetune_output_dir = r"C:\Users\guang\OneDrive\Desktop\IMDA\fine-tune-data\output"
            training_data = dataloader(input_dir = finetune_input_dir, output_dir = finetune_output_dir)
            OpenAIInferenceEngine.llm_agent = llm_model(finetune = False, training_data = training_data, openai_api_key=openai_api_key)
            self.openai_ai_model = load_OpenAI_ai_model()
            self.caller = self.openai_ai_model.get_image_information
        elif strategy == "PytorchCNN":
            from TorchInferenceEngine import load_torch_ai_model
            self.torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver2.pth')
            self.caller = self.torch_ai_model.get_image_information
        elif strategy == "Collaborative":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            from TorchInferenceEngine import load_torch_ai_model
            from CollaborativeInferenceEngine import load_collaborative_model
            OpenAIInferenceEngine.llm_agent = llm_model(openai_api_key=openai_api_key)
            openai_ai_model = load_OpenAI_ai_model()
            torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_model_ver2.pth')   
            self.collaborative_model = load_collaborative_model(torch_ai_model=torch_ai_model, openai_ai_model=openai_ai_model)
            self.caller = self.collaborative_model.get_image_information
        elif strategy == "Collaborative-with-NicheDiscriminator":
            import OpenAIInferenceEngine
            from OpenAIInferenceEngine import load_OpenAI_ai_model, llm_model
            from InferenceDataloader import dataloader
            from TorchInferenceEngine import load_torch_ai_model
            from CollaborativeInferenceEngine import load_collaborative_model
            OpenAIInferenceEngine.llm_agent = llm_model(openai_api_key=openai_api_key)
            openai_ai_model = load_OpenAI_ai_model()
            torch_ai_model = load_torch_ai_model(model_path = 'imda_technical_test_pytorch_O01IDiscriminator_model_ver3.pth')   
            self.collaborative_model = load_collaborative_model(torch_ai_model=torch_ai_model, openai_ai_model=openai_ai_model)
            self.caller = self.collaborative_model.get_image_information
        else:
            print(f"{strategy} is not a valid method")

    def __call__(self, im_path, save_path = None):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        result = self.caller(im_path)
        if save_path != None:
            with open(save_path, 'w') as sf:
                print(f"writing to {save_path}")
                sf.write(result['InferredCharacters'])
        return result