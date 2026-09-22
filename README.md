<h1 align="center">
  Few-Shot Demonstrations Elicit the Use of In-Context World Representations in LLMs
</h1>

<p align="center">
  Kohsei Matsutani,
  Gouki Minegishi,
  Core Francisco Park,
  Takeshi Kojima,
  Yusuke Iwasawa,
  Yutaka Matsuo
</p>

<p align="center">
  <a href="https://arxiv.org/abs/">
    <img src="https://img.shields.io/badge/arXiv-XXXXXX-b31b1b.svg" alt="arXiv">
  </a>
</p>

## Abstract
Large language models (LLMs), when acting as agents, are expected to take observed data in context, infer the latent state space underlying the world, and leverage it for downstream prediction. However, prior work demonstrated that LLMs struggle to use representations learned in context on a graph tracking task, where the model needs to construct a representation of the graph governing data generation process and use it for subsequent predictions. In this paper, we show that extending this to few-shot settings, where each demonstration is generated from a different world with either the same or different graph topologies, enhances its prediction on 6 models from 4 model families. To understand this improvement, we linearly probe a low-dimensional world representation that encodes graph information in the hidden states. Notably, we find that few-shot demonstrations relocate the world representation and increase its predictive use. Specifically, for each model, these world representations shift in directions nearly orthogonal to their original subspace, and interventions on these representations selectively impair performance more than interventions on other subspaces. Consistent with this insight, we show that few-shot demonstrations with observations from different worlds improve performance on ARC-AGI-1&2, web agent tasks, and Othello. Our findings elucidate the role and internal mechanisms of few-shot demonstrations in in-context world modeling. More broadly, our work advances our understanding of how LLM agents learn from in-context observations and provides implications for their further improvement.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

## Run


```bash
## Few-Shot Inference
bash scripts/infer.sh

## Activation Extraction
bash scripts/extract.sh

## Probing
bash scripts/probe.sh

## Intervention
bash scripts/intervention.sh
```

## Citation

```bibtex
@article{matsutani2026fewshot,
  title={Few-Shot Demonstrations Elicit the Use of In-Context World Representations in LLMs},
  author={Kohsei Matsutani, Gouki Minegishi, Core Francisco Park, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo},
  journal={arXiv preprint arXiv:},
  year={2026}
}
```
