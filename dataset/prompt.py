# prompt_config.py

PROMPT_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "coqa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Context: {story}\n"
            "Answer the question concisely (less than 8 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        ),
        "trivia_qa": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: {answer}"
        ),
        "sciq": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nA: "
        )
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
        )
    },
    "Qwen/Qwen2.5-7B-Instruct-1M": {
        "coqa": (
            "<|im_start|>user\n"
            "Context: {story}\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "trivia_qa": (
            "<|im_start|>user\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        ),
        "sciq": (
            "<|im_start|>user\n"
            "Answer the question concisely. (less than 5 words)\n"
            "Q: {question}<|im_end|>\n<|im_start|>assistant\nA: "
        )
    }
}

def get_prompt_template(model_name: str, dataset_name: str) -> str:
    """
    Retrieve the prompt template for a given model and dataset.
    """
    try:
        return PROMPT_TEMPLATES[model_name][dataset_name]
    except KeyError:
        raise ValueError(f"No prompt template found for model '{model_name}' and dataset '{dataset_name}'")
