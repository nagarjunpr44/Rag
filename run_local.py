from langchain_community.llms import CTransformers
import os

def initialize_llm():
    print("Loading local model...")
    local_model_path = os.path.abspath("model/llama-2-7b-chat.ggmlv3.q4_0.bin")
    llm = CTransformers(model= local_model_path,
                        model_type='llama',
                        config={'max_new_tokens': 725,
                                'temperature': 0.5,
                                "context_length": 5000}
                                )
    print("Local model loaded successfully.")
    return llm


if __name__ == '__main__':
    llm = initialize_llm()