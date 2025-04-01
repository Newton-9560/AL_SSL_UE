# prompt_config.py

PROMPT_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "coqa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Context: {story}\n"
            "Answer the question concisely. (less than 8 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "trivia_qa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "sciq": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "truthful_qa":(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 10 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "tydiqa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Context: {context}\n"
            "Extract less than 5 words answer from the context for the question.\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "ambig_qa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "squad": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Context: {context}\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "simple_qa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
    },
    "facebook/opt-6.7b": {
        "coqa": (
            "Context: {story}\n"
            "Answer the question concisely (less than 8 words).\n"
            "Q: {question}\nA:"
        ),
        "trivia_qa": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
        "sciq": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
        "truthful_qa":(
            "Answer the question concisely.\n"
            "Q: {question}\nA:"
        ),
        "tydiqa": (
            "Context: {context}\n"
            "Answer the question concisely (less than 5 words).\n"
            "Q: {question}\nA:"
        ),
        "ambig_qa": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
    },
    "Qwen/Qwen2.5-7B-Instruct-1M": {
        "coqa": (
            "<|im_start|>user\n"
            "Context: {story}\n"
            "Answer the question in eight words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "trivia_qa": (
            "<|im_start|>user\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "sciq": (
            "<|im_start|>user\n"
            "Answer the question in five words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "truthful_qa":(
            "<|im_start|>user\n"
            "Answer the question in ten words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "tydiqa": (
            "<|im_start|>user\n"
            "Context: {context}\n"
            "Extract less than 5 words answer from the context for the question.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "ambig_qa": (
            "<|im_start|>user\n"
            "Answer the question in five words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "squad": (
            "<|im_start|>user\n"
            "Context: {context}\n"
            "Answer the question in five words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "simple_qa": (
            "<|im_start|>user\n"
            "Answer the question in five words or less.\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        
    },
    "meta-llama/Llama-2-13b-chat-hf":{
        "coqa": (
            "Context: {story}\n"
            "Answer the question concisely (less than 8 words).\n"
            "Q: {question}\nA:"
        ),
        "trivia_qa": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
        "sciq": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
        "truthful_qa":(
            "Answer the question concisely.\n"
            "Q: {question}\nA:"
        ),
        "tydiqa": (
            "Context: {context}\n"
            "Answer the question concisely (less than 5 words).\n"
            "Q: {question}\nA:"
        ),
        "ambig_qa": (
            "Answer the question in five words or less.\n"
            "Q: {question}\nA:"
        ),
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "coqa": (
            "[INST] Context: {story} Answer the question in eight words or less: {question} [/INST]"
        ),
        "trivia_qa": (
            "[INST] Answer the question in five words or less: {question} [/INST]"
        ),
        "sciq": (
            "[INST] Answer the question in five words or less: {question} [/INST]"
        ),
        "truthful_qa":(
            "[INST] Answer the question in five words or less: {question} [/INST]"
        ),
        "tydiqa": (
            "[INST] Context: {context} Extract less than 5 words answer from the context for the question: {question} [/INST]"
        ),
        "ambig_qa": (
            "[INST] Answer the question in five words or less: {question} [/INST]"
        ),
        "squad": (
            "[INST] Context: {context} Answer the question in five words or less: {question} [/INST]"
        ),
        "simple_qa": (
           "[INST] Answer the question in five words or less: {question} [/INST]"
        ),
    },
}

def get_prompt_template(model_name: str, dataset_name: str) -> str:
    """
    Retrieve the prompt template for a given model and dataset.
    """
    try:
        return PROMPT_TEMPLATES[model_name][dataset_name]
    except KeyError:
        raise ValueError(f"No prompt template found for model '{model_name}' and dataset '{dataset_name}'")
