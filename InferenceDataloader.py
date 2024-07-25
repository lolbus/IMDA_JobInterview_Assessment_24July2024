import os
# A class to store data either to fine-tune openAI model by prompt engineering or for main inferencing.
class dataloader(object):
    def __init__(self, input_dir, output_dir, training_size = 99):
    # Initialize an empty dictionary to store the data
        data_dict = {}
        data_loaded = 0
        # Loop through all files in the input directory
        for filename in os.listdir(input_dir):
            
            if filename.endswith(".jpg"):
                # Get the full path of the image file
                img_path = os.path.join(input_dir, filename)
                
                # Construct the corresponding txt file name
                txt_filename = filename.replace(".jpg", ".txt")
                txt_filename = txt_filename.replace("input", "output")
                txt_path = os.path.join(output_dir, txt_filename)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as txt_file:
                        label = txt_file.read().strip()             
                        # Store the image path and label in the dictionary
                        data_dict[img_path] = label
                        data_loaded += 1
                        print(f"Pairing {filename} with {txt_filename}")
                        if data_loaded == training_size:
                            break
        self.data_dict = data_dict