# Sample Projects

A collection of self-contained research project prompts for screening prospective research collaborators. All projects use the [Pythia](https://arxiv.org/abs/2304.01373) model suite, which provides open models, open training data, and 143 intermediate checkpoints per model size.

## Projects

| Project | Description | Skills Tested |
|---------|-------------|---------------|
| [Predicting Memorization](predicting-memorization/) | Predict which training sequences a model will memorize | Feature engineering, supervised ML |
| [Circuit Analysis](circuit-analysis/) | Find and validate a computational circuit with TransformerLens | Mechanistic interpretability |
| [Data Attribution](data-attribution/) | Trace model behaviors back to influential training examples | Data attribution, experimental design |
| [Feature Acquisition](feature-acquisition/) | Track when linguistic features become decodable during training | Probing, experimental methodology |
| [Weight Statistics](weight-statistics/) | Find weight matrix statistics that predict capability emergence | Linear algebra, quantitative analysis |
| [Tokenizer Effects](tokenizer-effects/) | Systematically identify tokens that cause anomalous behavior | Exploratory analysis, creativity |

## Getting Started

Browse the projects above and pick one that interests you. Each project README explains the problem, provides starter code, and links to relevant papers. Projects are designed to be tackled independently.

All projects assume familiarity with Python and PyTorch. Most require a GPU for reasonable iteration speed, though smaller model sizes (70M-160M) can run on CPU in a pinch.

## Common Resources

- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
- **The Pile**: [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Gao et al., 2020)
- **Pythia paper**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
