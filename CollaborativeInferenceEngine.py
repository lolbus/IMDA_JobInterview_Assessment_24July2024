class load_collaborative_model(object):
    def __init__(self, torch_ai_model, openai_ai_model):
        self.openai_ai_model = openai_ai_model
        self.torch_ai_model = torch_ai_model
        self.openai_model_incapable_letters = ['0', 'O', '1', 'I']
    def get_image_information(self, input_image)->dict:
        """
        Main inference caller
        args: 
             input_image:PIL.Image.Image
        output: 
            dict containing inference result:dict
        """
        openai_output = self.openai_ai_model.get_image_information(input_image)
        torch_output = self.torch_ai_model.get_image_information(input_image)
        print(f"torch_output: {torch_output}")
        final_output = {'InferredCharacters':''}
        for i, predicted_char in enumerate(openai_output['InferredCharacters']):
            if predicted_char in self.openai_model_incapable_letters and torch_output['InferredCharacters'][i] in self.openai_model_incapable_letters:
                final_output['InferredCharacters'] +=  torch_output['InferredCharacters'][i]
            else:
                final_output['InferredCharacters'] += openai_output['InferredCharacters'][i]
        return final_output
        