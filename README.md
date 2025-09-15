[![PyPI version](https://badge.fury.io/py/litelines.svg)](https://badge.fury.io/py/litelines)
[![License: MIT](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/licenses/Apache2.0)

# litelines

Customize, control, and enhance LLM generation with logits processors, featuring visualization capabilities to inspect and understand state transitions.

## Installation

```bash
pip install litelines
```

## Dependencies

The only dependency is `outlines-core`.

## Supported Frameworks
* transformers
*

## Basic Usage

- Download a model and its tokenizer:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
```

- Define a logits processor through a Pydantic schema or a regular expression and visualize it:
```python
from litelines.transformers import SchemaProcessor

processor = SchemaProcessor(response_format=r"Yes\.|No\.", tokenizer=tokenizer)
processor.show_graph()
```
<img src="index_figures/Yes_or_No.jpg" />

- Generate a structured answer:
```python
messages = [{"role": "user", "content": "Are you sentient?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
generated = model.generate(**inputs, logits_processor=[processor])
print(tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:]))
# No.
```

- Visualize the selected path:
```python
processor.show_graph()
```
<img src="index_figures/Yes_or_No_selected_path.jpg" />

## 100% Guaranteed Valid JSON answer

- Define a pydantic schema describing the required JSON or provide the JSON schema as a string:
```python
from typing import Literal
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    """Correctly inferred `Sentiment` with all the required parameters with correct types."""
    label: Literal["positive", "negative"] = Field(
        ..., description="Sentiment of the text"
    )

'''
Alternatively, provide the JSON schema as a sting:
Sentiment = """{'description': 'Correctly inferred `Sentiment` with all the required parameters with correct types.',
 'properties': {'label': {'description': 'Sentiment of the text',
   'enum': ['positive', 'negative'],
   'title': 'Label',
   'type': 'string'}},
 'required': ['label'],
 'title': 'Sentiment',
 'type': 'object'}"""
'''
```

- Define the processor and visualize it:
```python
from litelines.transformers import SchemaProcessor

processor = SchemaProcessor(response_format=Sentiment, tokenizer=tokenizer)
processor.show_graph()
```
<img src="index_figures/Guaranteed_JSON.jpg" />

- Generate a structured answer:
```python
messages = [{"role": "user", "content": "What is the sentiment of the following text: 'Awesome'"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
generated = model.generate(**inputs, logits_processor=[processor])
print(tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:]))
# {"label": "positive"}
```

- Visualize the selected path:
```python
processor.show_graph()
```
<img src="index_figures/Guaranteed_JSON_selected_path.jpg" />


## 100% Guaranteed Valid Tool Calling answer

- Define a pydantic schema describing the tool:
```python
from typing import Literal
from pydantic import BaseModel, Field


class Sentiment(BaseModel):
    """Correctly inferred `Sentiment` with all the required parameters with correct types."""
    label: Literal["positive", "negative"] = Field(
        ..., description="Sentiment of the text"
    )
```

- Define the processor, add the parameter `include_tool_call=True` and visualize it:
```python
from litelines.transformers import SchemaProcessor

processor = SchemaProcessor(response_format=Sentiment, tokenizer=tokenizer, include_tool_call=True)
processor.show_graph()
```
<img src="index_figures/Guaranteed_Tool_Calling.jpg" />

- Generate a structured answer:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")

messages = [{"role": "user", "content": "What is the sentiment of the following text: 'Awesome'"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
generated = model.generate(**inputs, logits_processor=[processor])
print(tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:]))
# <tool_call>
# {"name": "Sentiment", "arguments": {"label": "positive"}}
# </tool_call>
```

- Visualize the selected path:
```python
processor.show_graph()
```
<img src="index_figures/Guaranteed_Tool_Calling_selected_path.jpg" />

## Allow Preamble but still get 100% Guaranteed Valid JSON/Tool Calling answer

- Define a pydantic schema describing the required JSON or provide the JSON schema as a string:
```python
from typing import Literal
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    """Correctly inferred `Sentiment` with all the required parameters with correct types."""
    label: Literal["positive", "negative"] = Field(
        ..., description="Sentiment of the text"
    )
```

- Define the processor, add parameter `allow_preamble=True` and visualize it:
```python
from litelines.transformers import SchemaProcessor

processor = SchemaProcessor(response_format=Sentiment, tokenizer=tokenizer, allow_preamble=True)
processor.show_graph()
```
<img src="index_figures/Guaranteed_JSON.jpg" />

- Generate a structured answer:
```python
messages = [{"role": "user", "content": "What is the sentiment of the following text: 'Awesome'"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
generated = model.generate(**inputs, logits_processor=[processor])
print(tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:]))
# The sentiment of the text "Awesome" is positive.
# {"label": "positive"}
```

- Visualize the selected path:
```python
processor.show_graph()
```
<img src="index_figures/Guaranteed_JSON_selected_path.jpg" />


