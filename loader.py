import numpy as np
from tqdm import tqdm
import math
import torch
def generate_data_and_labels(model, tokenizer, raw_data, layer=16, device='cuda'):
    """
    Generate training features and labels from the dataset using the provided model and tokenizer.

    Parameters:
        model: A model (e.g., from Hugging Face transformers) that outputs hidden states.
        tokenizer: The tokenizer corresponding to the model.
        layer (int): The index of the hidden layer from which to extract the hidden state.
        train_dataset (dict): A dictionary of training examples. Each key is an identifier and each value is a dictionary
                              that must contain at least the following keys:
                              - 'inputs': The input text.
                              - 'answer': The generated output text.
                              - 'align': A numerical alignment score.
                              Optionally, it may include:
                              - 'sar': Some additional score.

    Returns:
        train_data (np.ndarray): Array of feature vectors extracted from the specified hidden layer.
        train_label (list): A list of dictionaries with label and metadata information.
    """
    
    
    dataset = []
    labels = []
    
    # Iterate over the dataset keys with a progress bar.
    for id, data in tqdm(enumerate(raw_data)):
        if math.isnan(data['align']):
            continue
        # Concatenate the input and answer texts (adding a space between them).
        inp_text = data['inputs'] + data['answer'] + '<|eot_id|>'
        
        # Tokenize the concatenated text and move the tensors to the device.
        tokenized_input = tokenizer(inp_text, return_tensors="pt").to(device)
        
        # Run the model with output_hidden_states=True.
        with torch.no_grad():
            outputs = model(input_ids=tokenized_input['input_ids'], output_hidden_states=True)
        
        # Extract the hidden state from the specified layer.
        # We assume the output has a "hidden_states" attribute, which is a tuple of tensors.
        # Here, we select the first (and only) batch element and then take the hidden state for the last token.
        hidden_state = outputs.hidden_states[layer][0][-1].cpu().detach().numpy()
        dataset.append(hidden_state)
        
        # Build the label dictionary.
        # If 'sar' is not guaranteed to be in the data, we use .get() to safely access it.
        label = {
            'align_score': data['align'],
            'SAR': data.get('sar', None),
            'input_text': data['inputs'],
            'output_text': data['answer'],
            'idx': id
        }
        labels.append(label)
        torch.cuda.empty_cache()
    
    # Convert the list of feature vectors to a NumPy array.
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

# Example usage:
# train_data, train_label = generate_train_data_and_labels(model, tokenizer, 16, train_dataset)
