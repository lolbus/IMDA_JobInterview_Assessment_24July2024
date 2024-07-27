import os
import langchain
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
import base64
import json
from langchain.tools import tool

# Prompt Engineering portion

vision_prompt = """Given the Captcha image, where 
They resemble each other very much where the identify that the texture, nature of the font, spacing of the font, morphological characteristic of the letters and numerals arev very consistent.
Infer the image's characters (Only capital letters and numbers)

"""

few_shot_prompt = """Few shot prompting fine tunning: it is very important to take note of the difference between:
0 and O  

and

1 and I

In our dataset, their difference are consistent ('O' being more circular than '0' and having more rounding pixels at the top and bottom tip of the character and 'I' having a longer flat surface at the tip of the character than '1') and you must fine-tune yourself to adapt to the font format.
Pay attention to the top tip of the character when trying to distinguish these four characters.
A list of sample data describing the difference between '0', 'O', '1' and 'I' are provided to allow you to fine-tune your performance for the given task: 


"""

class llm_model(object):
    def __init__(self, finetune = False, training_data = None, openai_api_key = "<INSERT_OPENAI_API_KEY_HERE>"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0, model="gpt-4o", max_tokens=1024)
        self.format_instruction = format_instruction = """
             Provide your output in the format of a python dict object code {"InferredCharacters": "XXXXX"}.
             Do not provide any other information. 
             Do not provide the backticks or any characters not related to the dict string such that I can convert to dict object directly by json.load(result)"""
        self.vision_prompt = vision_prompt 
        if finetune:
            self.vision_prompt = self.vision_prompt + few_shot_prompt
            for data_path in training_data.data_dict.keys():
                encoded_image_data = encode_image(data_path)
                fine_tune_data = f"""```Encoded image data: {encoded_image_data} -> True label: {training_data.data_dict[data_path]}``` \n"""
                self.vision_prompt = self.vision_prompt + fine_tune_data

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# Declare an empty llm agent
llm_agent = llm_model()
            
def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
    image_base64 = encode_image(image_path)
    return {"image": image_base64}

load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)



@chain
def image_model(inputs: dict) -> str:
    """Invoke model with image and prompt."""
    model = llm_agent.llm
    # print("Final vision prompt\n", inputs["prompt"])
    msg = model.invoke(
             [SystemMessage(content=llm_agent.format_instruction),
              HumanMessage(
             content=[
             {"type": "text", "text": inputs["prompt"]},
             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{inputs['image']}"}},
             ])]
             )
    return msg.content

@tool
def convert_output_to_dict(llm_output:str) -> dict:
    """Convert the output of the LLM to a simple dict with only the predicted output as the key"""
    try:
        response_dict = json.loads(llm_output)
    except json.JSONDecodeError:
        response_dict = {"error": "Invalid JSON response"}
    return response_dict


#OpenAI Inference Engine
class load_OpenAI_ai_model(object):
    def __init__(self):
        self.llm_agent = llm_agent

    def get_image_information(self, image_path: str) -> dict:
        vision_prompt = llm_agent.vision_prompt
        vision_chain = load_image_chain | image_model | convert_output_to_dict
        return vision_chain.invoke({'image_path': f'{image_path}', 
                                   'prompt': vision_prompt})
