# NLI-BERT_Cases

## Natural Language Inference Classification with Hugging Face BERT

This repository contains a Natural Language Inference (NLI) classification model that leverages the power of the Hugging Face BERT model. The model is trained on the [FarsTail dataset](https://github.com/dml-qom/FarsTail) for Persian language.

## Overview

Natural Language Inference (NLI) is a task in natural language processing that involves determining the relationship between two sentences: "entailment", "contradiction", or "neutral". This repository provides a fine-tuned BERT-based model to perform NLI classification for Persian language.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers library

You can install the required packages using the following command:

install pytorh according to https://pytorch.org/

```bash
pip install transformers
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/NLI-BERT_Cases.git
```

2. Download the FarsTail dataset from [here](https://github.com/dml-qom/FarsTail) and place it in the `data/` directory.

3. Train the model:

```bash
python train.py
```

4. Evaluate the model:

```bash
python evaluate.py
```

## Model Fine-tuning

The model is fine-tuned on the FarsTail dataset using the Hugging Face Transformers library. You can find the fine-tuning script in the `train.py` file. You can further fine-tune the model by adjusting hyperparameters or using additional data.

## Results

After training, the model achieves an accuracy of X% on the test set.

## Pre-trained Model

You can download the pre-trained model from [here](link-to-pretrained-model).

## Citation

If you use this code or the FarsTail dataset in your research, please cite:

```
@article{fars_tail_paper,
  title={Title of the FarsTail Paper},
  author={Author Names},
  journal={Journal Name},
  year={Year},
  url={Link to the paper},
  code={Link to the repository},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the creators of the FarsTail dataset for providing valuable data for this project.

---

Feel free to contribute to this repository by opening issues or creating pull requests. Happy classifying!

For any questions or inquiries, please contact [Your Name](mailto:iyasiniyasin98@gmail.com).