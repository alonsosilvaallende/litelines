# Getting started

This guide will walk you through the basics of using Litelines to get structured generation from language models. By the end, you'll understand how to:

1. Install Litelines
2. Generate a basic structured response
3. Generate a basic streamed structured response


## Installation

To install `litelines`:

=== "pip"

    ``` sh
    pip install litelines
    ```

=== "uv"

    ``` sh
    uv pip install litelines
    ```

## Your First Structured Generation

Let's start with a simple example.

### Download a model and its tokenizer
=== "transformers"

    ``` python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = torch.device("cuda") # "cuda", "mps", or "cpu"
    
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ```

=== "vllm"

### Prepare the inputs to the LLM

=== "transformers"

    ``` python
    user_input = "What is the sentiment of the following text: 'Awesome'"
    messages = [{"role": "user", "content": user_input}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True
    ).to(model.device)
    ```

=== "vllm"



### Define a Pydantic schema describing the required JSON

=== "transformers"

    ``` python
    from typing import Literal
    from pydantic import BaseModel, Field
    
    class Sentiment(BaseModel):
        """Correctly inferred `Sentiment`."""
        label: Literal["positive", "negative"] = Field(
            ..., description="Sentiment of the text"
        )
    ```

=== "vllm"

### Define the processor and visualize it

=== "transformers"

    ``` python
    from litelines.transformers import SchemaProcessor
    
    processor = SchemaProcessor(response_format=Sentiment, tokenizer=tokenizer)
    processor.show_graph()
    ```

=== "vllm"

### Generate a structured response

=== "transformers"

    ``` python
    generated = model.generate(**inputs, logits_processor=[processor])
    print(tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:]))
    # {"label": "positive"}
    ```

=== "vllm"


### Visualize the selected path

=== "transformers"

    ``` python
    processor.show_graph()
    ```

=== "vllm"


## Your First Streamed Structured Generation

Since Litelines gives you the processor, you can do whatever you want with it. In particular, you can generate a streaming response like you would normally do (just don't forget to add the processor).

=== "transformers"

    ``` python
    from threading import Thread
    from transformers import TextIteratorStreamer
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        inputs, streamer=streamer, logits_processor=[processor], max_new_tokens=100
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    assistant_response = ""
    for chunk in streamer:
        if tokenizer.eos_token in chunk or tokenizer.pad_token in chunk:
            chunk = chunk.split(tokenizer.eos_token)[0]
            chunk = chunk.split(tokenizer.pad_token)[0]
        assistant_response += chunk
        print(chunk, end="")
    
    thread.join()
    ```

=== "vllm"
