# IMDA Technical Assessment July 2024 for Candidate Teng Guang Way, Wayne

## Requirements to run the code
To install the necessary packages, run the following pip command in a **Python 3.11.9** environment:

    pip install langchain
    pip install langchain_openai
    pip install pillow
    pip3 install torch torchvision torchaudio

You must also attain an **OpenAI API key** for some of the solutions that uses OpenAI's LLM to infer the Captcha image. 

## How to run the final solutions
To setup the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.

To run the **final solution**, follow these steps:

1. Open the **main_AssessmentUsage_showcase.ipynb** using Jupyter Notebook
2. Run the cells in the notebook sequentially to execute the code.

or, run the following python command after navigating to the project directory:

    openai_api_key = "<OPENAI_APY_KEY>"
    from CaptchaClass import Captcha
    Captcha_inferencer = Captcha(strategy = "Collaborative-WORB", openai_api_key=openai_api_key)
    Captcha_inferencer(r".\sampleCaptchas\input\input100.jpg", save_path = ".\sample_input100.txt")

or, if without **OpenAI API key**, run the less robust solution, with the following python command after navigating to the project directory:

    from CaptchaClass import Captcha
    Captcha_inferencer = Captcha(strategy = "PytorchCNN-WORB")
    Captcha_inferencer(r".\sampleCaptchas\input\input100.jpg", save_path = ".\sample_input100.txt")

You may set the strategy to a **non final** one which are all relevant iterative steps to achieve the **final solution**. Available strategies,

Available Strategies:
1. OpenAI GPT4o AI Assistant (Requires OpenAI API Key:Yes) - ```LLMAIAssistant```
2. PyTorch model self trained model (Requires OpenAI API Key:No) - ```PytorchCNN```
3. Pytorch model + OpenAI GPT4o AI Assistant Collaboration (Requires OpenAI API Key:Yes) - ```Collaborative```
4. Pytorch model trained as O, 0, 1, I 4-class Discriminator + OpenAI GPT4o AI Assistant Collaboration (Requires OpenAI API Key:Yes) - ```Collaborative-with-NicheDiscriminator```
5. PyTorch model self trained model with White OR Black pixels conversion (Requires OpenAI API Key:No) - ```PytorchCNN-WORB```
6. Pytorch model + OpenAI GPT4o AI Assistant Collaboration (Requires OpenAI API Key:Yes) (Favourite Final Solution) - ```Collaborative-WORB```

## Explanation of solution strategies

### 1.OpenAI GPT4o AI Assistant
This is the first draft of my solution. It uses a LLM AI Assistant. In this example, we pick GPT-4o which is the most powerful publicly available LLM available for the public. 
GPT-4o is integrated with a ViT model for vision task. The ViT model was trained on a huge dataset and is capable of handle a wide range of classification tasks, including the relevant one in this assessment without any fine-tunning.
Unfortunately, it struggles to tell the difference between "O", "0", "1", "I" for this specific captcha image's font. Incorporating few-shot prompting by prompt engineering didn't help much either. 
This solution is good as it does not requires AI expertise to incorporates, any decently experienced Software Engineer may use the high-level GenAI frameworks available as of July 2024 to incorporate the method as part of the solution with best use practice.
However developer must top up credits in their OpenAI account and provide an OpenAI API key that will continuously bill the account for usage. The cost can be moderated by using a cheaper and weak model if use case allows.

### 2.PyTorch model self trained model
This is the second draft of my solution. It uses a self-trained SimpleCNN model that crop images into 5 equal parts containing the characters to infer, and infer the character sequentially as a multi-class classification task. 
This method may not work if the characters in the unseen Captcha images are skewed, distorted, displaced differently as the cropping method used might not be able to handle. 
This method uses the tiny dataset (25 instances of Captcha images) provided to train a CNN model, it is able to handle the difference between "O", "0", "1", "I" easily, thanks to the short range dependency mechanism of CNN model to pay attention to edge details of the character. 
But some obvious characters like Y gets mistaken as V sometimes, even though from distance, any person can tell the difference of the 2 characters. This performance issue may be solved if a bigger dataset is provided. 
Since it is able to handle the characters which the ViT incorporated inside GPT4o failed, in strategy 3, and 4, collaboration methods was developed, leveraging the more robust ViT Transformer model and the leanly trained CNN model for special specific case to work together.

### 3.Pytorch model + OpenAI GPT4o AI Assistant Collaboration 
This is the third draft and also the final solution draft. It collaborates the 2 models developed in **Method 1** and **Method 2**. 
Through this model, first the Captcha image will be passed to the OpenAI GPT4o AI Assistant for inference with its ViT model, 
then the output are checked for any illegal characters, namely ['O', '0', 'I', '1'], which the AI Assistant's in-built ViT has difficulties distinguishing,
if an illegal character is found on a specific position, the output of the Pytorch Model, will infer that character and make the final call on the character (Deciding whether it is "1 or I" or "O or 0").
All other acceptable characters inferred by the AI Assistant will take credit from the AI Assistant's inference.

### 4.Pytorch model trained as O, 0, 1, I 4-class Discriminator + OpenAI GPT4o AI Assistant Collaboration
This is the last draft made. Like 3., it collaborates the 2 model developed in **Method 1** and **Method 2** The key difference would be this strategy uses a more advanced methodology to train the PyTorch model.
During the training phase, first a PyTorch model was trained like in **Method 2**, then the model was loaded as pre-trained model onto another PyTorch trainer, and fine-tune on a character dataset compromises of ['0', 'O', '1', 'I'] characters.
As the dataset provided was already extremely sparse, selectively using only the most relevant data to fine-tune, result in very negligible to the model, and there are no obvious indication, that
the model improved from utilizing the transfer-learning principle.

### 5.PyTorch model self trained model with White OR Black pixels conversion
This method uses a newly developed function to preprocess the image data before inference with or training of a PytorchCNN model. The function was developed by rounding all pixels below 128 to 0 and all pixels above 128 to 255.
However the robustness of the function may be uncertain as the conversion method was applied and evaluated on a tiny dataset. 
Even though it is possible to infer the entire dataset (combination of sparse amount of seen/unseen images), I would not certain enough trust the model completely for deployment. 
However if the developed image conversion function works, then the nature of problem has been simplified significantly for a Simple CNN to solve. 
The conversion effect removes the greyish zig zag lines from the captcha image effectively. Below diagram demonstrate the conversion effects. 
Apart from the addition of a new preprocess before inputing an inference/training data. The unprocessed original Captcha image was also used as additional training data. 
Everything else is the same as in **Method 2**.

<img src="https://raw.githubusercontent.com/lolbus/IMDA_JobInterview_Assessment_24July2024/main/WORB_effect.PNG" alt="Demo of WORB conversion.png" border="1" /><br>

### 6.Pytorch model self trained model with White OR Black pixels conversion + OpenAI GPT4o AI Assistant Collaboration **(Best, favourite chosen final solution)**
This method uses the PyTorch model developed in **Method 5** to infer 'O', '0', '1', 'I' like the case **Method 3** uses **Method 2**. 
As the most of the inferences rely on the huge ViT model in-built within GPT-4o, this is the more trustable model relative to **Method 5.** after greatly simplifying the solution.

## Strengths and Weaknesses of each strategy
Below is a diagram benchmarking the solutions based on Performance (Approximate Robustness), Development Labor, Development Simplicity, Deployment cost.


| Strategy                                                                                                 | Performance (Approximate robustness) | Development Labor | Development Simplicity | Deployment cost per use |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------- | ---------------------- | --------------------- |
| 1.OpenAI GPT4o AI Assistant                                                                                | ⭐⭐⭐                                 | ⭐⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐                  | ⭐⭐⭐                  |
| 2.PyTorch model self trained model                                                                         | ⭐⭐                                  | ⭐⭐⭐⭐              | ⭐⭐⭐⭐                   | ⭐⭐⭐⭐⭐                |
| 3.Pytorch model + OpenAI GPT4o AI Assistant Collaboration                                                  | ⭐⭐⭐⭐⭐                               | ⭐⭐⭐               | ⭐⭐⭐⭐                   | ⭐⭐⭐                  |
| 4.Pytorch model trained as O, 0, 1, I 4-class Discriminator + OpenAI GPT4o AI Assistant Collaboration      | ⭐⭐⭐⭐⭐                               | ⭐⭐⭐               | ⭐⭐                     | ⭐⭐⭐                  |
| 5.PyTorch model self trained model with White OR Black pixels conversion                                   | ⭐⭐⭐⭐                                | ⭐⭐⭐               | ⭐⭐⭐                     | ⭐⭐⭐⭐⭐                  | 
| 6.Pytorch model self trained model with White OR Black pixels conversion + OpenAI GPT4o AI Assistant Collaboration  | ⭐⭐⭐⭐⭐⭐                               | ⭐⭐               | ⭐⭐⭐                     | ⭐⭐⭐                  |

## Acknowledgments
- LangChain, OpenAI, PyTorch
