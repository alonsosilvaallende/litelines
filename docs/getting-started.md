# Getting started

This guide will walk you through the basics of using Litelines to get structured generation from language models. By the end, you'll understand how to:

1. Install Litelines
2. Generate a basic structured response
3. Generate a basic streamed structured response


## Installation

To install Litelines:

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

```graphviz dot attack_plan0.svg
// Allowed Transitions Graph
digraph {
	graph [label="Allowed Paths
Regular expression: [\\n\\t ]*\\{[\\n\\t ]*\"label\"[\\n\\t ]*:[\\n\\t ]*(\"positive\"|\"negative\")[\\n\\t ]*\\}",labelloc="t",labeljust="l"]
	rankdir=LR;size="None";ratio=None;
	0 [label="0" shape=circle]
	1 [label="1" shape=circle]
	2 [label="2" shape=circle]
	3 [label="3" shape=circle]
	4 [label="4" shape=circle]
	5 [label="5" shape=circle]
	6 [label="6" shape=circle]
	7 [label="7" shape=circle]
	8 [label="8" shape=circle]
	9 [label="9" shape=circle]
	10 [label="10" shape=circle]
	11 [label="11" shape=circle]
	12 [label="12" shape=circle]
	13 [label="13" shape=circle]
	14 [label="14" shape=circle]
	15 [label="15" shape=circle]
	16 [label="16" shape=circle]
	17 [label="17" shape=circle]
	18 [label="18" shape=circle]
	19 [label="19" shape=circle]
	20 [label="20" shape=circle]
	21 [label="21" shape=circle]
	22 [label="22" shape=circle]
	23 [label="23" shape=circle]
	24 [label="24" shape=circle]
	25 [label="25" shape=circle]
	26 [label="26" shape=circle]
	27 [label="27" shape=circle]
	28 [label="28" shape=doublecircle]
	node [shape=none]
	"" [label=""]
	"" -> 0
	0 -> 0 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	0 -> 1 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">53632</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">1476</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">90</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	0 -> 23 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">4913</font></td><td>{"</td></tr><tr><td align="right"><font color="#00b4d8">5212</font></td><td>{"</td></tr></table>>]
	1 -> 1 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	1 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">92667</font></td><td>"label</td></tr></table>>]
	1 -> 23 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">330</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr></table>>]
	2 -> 3 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1837</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	2 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">788</font></td><td>":</td></tr><tr><td align="right"><font color="#00b4d8">51418</font></td><td>":</td></tr><tr><td align="right"><font color="#00b4d8">4660</font></td><td>":</td></tr></table>>]
	2 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">3252</font></td><td>":"</td></tr></table>>]
	3 -> 3 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	3 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">25</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">47446</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">549</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	3 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">2974</font></td><td>:"</td></tr><tr><td align="right"><font color="#00b4d8">34638</font></td><td>:"</td></tr></table>>]
	4 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	4 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">330</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr></table>>]
	5 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">2724</font></td><td>posit</td></tr></table>>]
	5 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">30487</font></td><td>positive</td></tr></table>>]
	5 -> 11 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">77</font></td><td>n</td></tr></table>>]
	5 -> 12 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">811</font></td><td>ne</td></tr></table>>]
	5 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">28775</font></td><td>neg</td></tr></table>>]
	5 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">42224</font></td><td>negative</td></tr></table>>]
	5 -> 19 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">79</font></td><td>p</td></tr></table>>]
	5 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">966</font></td><td>pos</td></tr></table>>]
	5 -> 22 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">5368</font></td><td>po</td></tr></table>>]
	6 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	6 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">533</font></td><td>ive</td></tr></table>>]
	6 -> 10 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">344</font></td><td>iv</td></tr></table>>]
	7 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">586</font></td><td>ve</td></tr></table>>]
	7 -> 10 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">85</font></td><td>v</td></tr></table>>]
	8 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	8 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9207</font></td><td>"}</td></tr></table>>]
	9 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	9 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">92</font></td><td>}</td></tr><tr><td align="right"><font color="#00b4d8">335</font></td><td>}</td></tr></table>>]
	10 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	11 -> 12 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	11 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">791</font></td><td>eg</td></tr></table>>]
	11 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">11188</font></td><td>ega</td></tr></table>>]
	11 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15060</font></td><td>egative</td></tr></table>>]
	12 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">70</font></td><td>g</td></tr></table>>]
	12 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">6743</font></td><td>ga</td></tr></table>>]
	13 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">64</font></td><td>a</td></tr></table>>]
	13 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9307</font></td><td>ati</td></tr></table>>]
	13 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1388</font></td><td>ative</td></tr></table>>]
	13 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">19488</font></td><td>ativ</td></tr></table>>]
	13 -> 18 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">266</font></td><td>at</td></tr></table>>]
	14 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">10251</font></td><td>ti</td></tr></table>>]
	14 -> 18 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">83</font></td><td>t</td></tr></table>>]
	15 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">586</font></td><td>ve</td></tr></table>>]
	15 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">85</font></td><td>v</td></tr></table>>]
	16 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	16 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9207</font></td><td>"}</td></tr></table>>]
	17 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	18 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	18 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">533</font></td><td>ive</td></tr></table>>]
	18 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">344</font></td><td>iv</td></tr></table>>]
	19 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">34054</font></td><td>osit</td></tr></table>>]
	19 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">436</font></td><td>os</td></tr></table>>]
	19 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">30724</font></td><td>osi</td></tr></table>>]
	19 -> 22 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">78</font></td><td>o</td></tr></table>>]
	20 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">275</font></td><td>it</td></tr></table>>]
	20 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">12303</font></td><td>iti</td></tr></table>>]
	20 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">3404</font></td><td>itive</td></tr></table>>]
	20 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	21 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">83</font></td><td>t</td></tr></table>>]
	21 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">10251</font></td><td>ti</td></tr></table>>]
	22 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">46865</font></td><td>sit</td></tr></table>>]
	22 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">82</font></td><td>s</td></tr></table>>]
	22 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">6321</font></td><td>si</td></tr></table>>]
	23 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1502</font></td><td>label</td></tr></table>>]
	23 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">14380</font></td><td>lab</td></tr></table>>]
	23 -> 26 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">75</font></td><td>l</td></tr></table>>]
	23 -> 27 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">4260</font></td><td>la</td></tr></table>>]
	24 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">301</font></td><td>el</td></tr></table>>]
	24 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	25 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">75</font></td><td>l</td></tr></table>>]
	26 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">780</font></td><td>abel</td></tr></table>>]
	26 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">370</font></td><td>ab</td></tr></table>>]
	26 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">8229</font></td><td>abe</td></tr></table>>]
	26 -> 27 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">64</font></td><td>a</td></tr></table>>]
	27 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9779</font></td><td>bel</td></tr></table>>]
	27 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">65</font></td><td>b</td></tr></table>>]
	27 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1371</font></td><td>be</td></tr></table>>]
}
```

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


```graphviz dot attack_plan1.svg
// Allowed Transitions Graph
digraph {
	graph [label="Allowed Paths
Regular expression: [\\n\\t ]*\\{[\\n\\t ]*\"label\"[\\n\\t ]*:[\\n\\t ]*(\"positive\"|\"negative\")[\\n\\t ]*\\}",labelloc="t",labeljust="l"]
	rankdir=LR;size="None";ratio=None;
	0 [label="0" shape=circle]
	1 [label="1" shape=circle]
	2 [label="2" shape=circle]
	3 [label="3" shape=circle]
	4 [label="4" shape=circle]
	5 [label="5" shape=circle]
	6 [label="6" shape=circle]
	7 [label="7" shape=circle]
	8 [label="8" shape=circle]
	9 [label="9" shape=circle]
	10 [label="10" shape=circle]
	11 [label="11" shape=circle]
	12 [label="12" shape=circle]
	13 [label="13" shape=circle]
	14 [label="14" shape=circle]
	15 [label="15" shape=circle]
	16 [label="16" shape=circle]
	17 [label="17" shape=circle]
	18 [label="18" shape=circle]
	19 [label="19" shape=circle]
	20 [label="20" shape=circle]
	21 [label="21" shape=circle]
	22 [label="22" shape=circle]
	23 [label="23" shape=circle]
	24 [label="24" shape=circle]
	25 [label="25" shape=circle]
	26 [label="26" shape=circle]
	27 [label="27" shape=circle]
	28 [label="28" shape=doublecircle]
	node [shape=none]
	"" [label=""]
	"" -> 0
	0 -> 0 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	0 -> 1 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">53632</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">1476</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">90</font></td><td>{</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>> color=red penwidth=3.0]
	0 -> 23 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">4913</font></td><td>{"</td></tr><tr><td align="right"><font color="#00b4d8">5212</font></td><td>{"</td></tr></table>>]
	1 -> 1 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	1 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">92667</font></td><td>"label</td></tr></table>>]
	1 -> 23 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">330</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr></table>> color=red penwidth=3.0]
	2 -> 3 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1837</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	2 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">788</font></td><td>":</td></tr><tr><td align="right"><font color="#00b4d8">51418</font></td><td>":</td></tr><tr><td align="right"><font color="#00b4d8">4660</font></td><td>":</td></tr></table>> color=red penwidth=3.0]
	2 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">3252</font></td><td>":"</td></tr></table>>]
	3 -> 3 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	3 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">25</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">47446</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">549</font></td><td>:</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	3 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">2974</font></td><td>:"</td></tr><tr><td align="right"><font color="#00b4d8">34638</font></td><td>:"</td></tr></table>>]
	4 -> 4 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	4 -> 5 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">330</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr></table>> color=red penwidth=3.0]
	5 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">2724</font></td><td>posit</td></tr></table>>]
	5 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">30487</font></td><td>positive</td></tr></table>> color=red penwidth=3.0]
	5 -> 11 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">77</font></td><td>n</td></tr></table>>]
	5 -> 12 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">811</font></td><td>ne</td></tr></table>>]
	5 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">28775</font></td><td>neg</td></tr></table>>]
	5 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">42224</font></td><td>negative</td></tr></table>>]
	5 -> 19 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">79</font></td><td>p</td></tr></table>>]
	5 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">966</font></td><td>pos</td></tr></table>>]
	5 -> 22 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">5368</font></td><td>po</td></tr></table>>]
	6 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	6 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">533</font></td><td>ive</td></tr></table>>]
	6 -> 10 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">344</font></td><td>iv</td></tr></table>>]
	7 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">586</font></td><td>ve</td></tr></table>>]
	7 -> 10 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">85</font></td><td>v</td></tr></table>>]
	8 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>> color=red penwidth=3.0]
	8 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9207</font></td><td>"}</td></tr></table>>]
	9 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">23459</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">59101</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">503</font></td><td></td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	9 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">92</font></td><td>}</td></tr><tr><td align="right"><font color="#00b4d8">335</font></td><td>}</td></tr></table>> color=red penwidth=3.0]
	10 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	11 -> 12 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	11 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">791</font></td><td>eg</td></tr></table>>]
	11 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">11188</font></td><td>ega</td></tr></table>>]
	11 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15060</font></td><td>egative</td></tr></table>>]
	12 -> 13 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">70</font></td><td>g</td></tr></table>>]
	12 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">6743</font></td><td>ga</td></tr></table>>]
	13 -> 14 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">64</font></td><td>a</td></tr></table>>]
	13 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9307</font></td><td>ati</td></tr></table>>]
	13 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1388</font></td><td>ative</td></tr></table>>]
	13 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">19488</font></td><td>ativ</td></tr></table>>]
	13 -> 18 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">266</font></td><td>at</td></tr></table>>]
	14 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">10251</font></td><td>ti</td></tr></table>>]
	14 -> 18 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">83</font></td><td>t</td></tr></table>>]
	15 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">586</font></td><td>ve</td></tr></table>>]
	15 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">85</font></td><td>v</td></tr></table>>]
	16 -> 9 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">15620</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">698</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">1</font></td><td>"</td></tr><tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr></table>>]
	16 -> 28 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9207</font></td><td>"}</td></tr></table>>]
	17 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	18 -> 15 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	18 -> 16 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">533</font></td><td>ive</td></tr></table>>]
	18 -> 17 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">344</font></td><td>iv</td></tr></table>>]
	19 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">34054</font></td><td>osit</td></tr></table>>]
	19 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">436</font></td><td>os</td></tr></table>>]
	19 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">30724</font></td><td>osi</td></tr></table>>]
	19 -> 22 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">78</font></td><td>o</td></tr></table>>]
	20 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">275</font></td><td>it</td></tr></table>>]
	20 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">12303</font></td><td>iti</td></tr></table>>]
	20 -> 8 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">3404</font></td><td>itive</td></tr></table>>]
	20 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">72</font></td><td>i</td></tr></table>>]
	21 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">83</font></td><td>t</td></tr></table>>]
	21 -> 7 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">10251</font></td><td>ti</td></tr></table>>]
	22 -> 6 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">46865</font></td><td>sit</td></tr></table>>]
	22 -> 20 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">82</font></td><td>s</td></tr></table>>]
	22 -> 21 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">6321</font></td><td>si</td></tr></table>>]
	23 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1502</font></td><td>label</td></tr></table>> color=red penwidth=3.0]
	23 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">14380</font></td><td>lab</td></tr></table>>]
	23 -> 26 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">75</font></td><td>l</td></tr></table>>]
	23 -> 27 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">4260</font></td><td>la</td></tr></table>>]
	24 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">301</font></td><td>el</td></tr></table>>]
	24 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">68</font></td><td>e</td></tr></table>>]
	25 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">75</font></td><td>l</td></tr></table>>]
	26 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">780</font></td><td>abel</td></tr></table>>]
	26 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">370</font></td><td>ab</td></tr></table>>]
	26 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">8229</font></td><td>abe</td></tr></table>>]
	26 -> 27 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">64</font></td><td>a</td></tr></table>>]
	27 -> 2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">9779</font></td><td>bel</td></tr></table>>]
	27 -> 24 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">65</font></td><td>b</td></tr></table>>]
	27 -> 25 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr><tr><td align="right"><font color="#00b4d8">1371</font></td><td>be</td></tr></table>>]
}
```

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
