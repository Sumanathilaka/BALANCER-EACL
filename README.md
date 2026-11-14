# BALANCER: Balanced Agentic Learning via Adaptive dyNamic Context ExtRaction for Hard Word Sense Disambiguation

This repository contains experiments and data for English and multilingual word sense
disambiguation (WSD). It includes notebooks for generating or enriching
sense-specific examples with Google ADK agents, example collections for
English and German, a prompt template, and scripts for further fine-tuning
Gemma and Qwen models.

## Repository contents

| Path | Description |
| --- | --- |
| `Balancer.ipynb` | Agent-based workflow for filtering candidate senses, searching for illustrative examples, validating them, and preparing prompts from the `deshanksuman/42D_sensemapped` Hugging Face dataset. |
| `Balancer_without_agentic.ipynb` | A simpler example-generation workflow. Replace the placeholder dataset name in the notebook with the Hugging Face dataset you want to use. |
| `prompt_WSF.txt` | Example German WSD prompt using sense IDs and example sentences. |
| `42D/42D_examples.txt` | English sense/example data. |
| `HardEN/hardenExample.txt` | English examples for the HardEN collection. |
| `HardMWSD/exampleMWSD.txt` | German examples for multilingual WSD. |
| `Model Finetuning/finetune_Gemma4B_Phase2.py` | Fine-tuning workflow based on Gemma 3 4B and a previously saved PEFT adapter. |
| `Model Finetuning/finetune_Qwen4b_Phase2.py` | Fine-tuning workflow based on Qwen3 4B and a previously saved PEFT adapter. |

The example text files contain sense IDs associated with example sentences.
The notebook workflows can produce an `examples.json` file containing
generated prompts and gold sense labels.

## Requirements

- Python and Jupyter Notebook or JupyterLab for the notebooks.
- Google ADK and the Google GenAI SDK for the agent-based notebook. The
  notebook uses the `gemini-2.5-flash-lite` model and Google Search.
- Hugging Face `datasets`, `transformers`, `trl`, `peft`, and PyTorch for the
  fine-tuning scripts.
- A CUDA-capable GPU is recommended for fine-tuning 4B-parameter models.
- Access to the referenced Hugging Face datasets, models, and adapters. Some
  model repositories may require accepting their terms or authenticating.

This repository does not currently include a dependency lockfile or pinned
package versions. Install compatible versions of the packages required by the
workflow you intend to run.

## Setup and usage

1. Clone the repository and open it in Jupyter or your preferred Python
   environment.
2. Install the dependencies for the notebook or script you plan to use.
3. Configure authentication for the external services:
   - Set `GOOGLE_API_KEY` securely before running the Google ADK notebook.
   - The fine-tuning scripts currently contain empty Hugging Face token
     placeholders. Configure them to use a secure authentication method (for
     example, the Hugging Face CLI login and cached credentials) before
     loading gated models or publishing adapters; do not commit tokens.
4. Open the relevant notebook and run its cells in order, or run a fine-tuning
   script in an environment with the required model access and GPU resources.
5. Review the dataset and model identifiers in the notebook or script before
   starting a run. In particular, `Balancer_without_agentic.ipynb` uses
   `dataset_name` as a placeholder.

The fine-tuning scripts load an existing adapter, train it on
`deshanksuman/senseMapped_MWSD_Train`, save adapter files locally under
`lora_adapters/`, and are configured to publish the resulting adapter to the
Hugging Face Hub. Review the target repository and authentication settings
before enabling publication.

## Credentials and generated files

Training checkpoints and generated files such as `examples.json` and
`lora_adapters/` are created when workflows are run; they are not included in
the repository by default.


## Publication

Accepted to EACL 2027 findings.
Paper title : BALANCER: Balanced Agentic Learning via Adaptive dyNamic Context ExtRaction for Hard Word Sense Disambiguation
Authors: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough
