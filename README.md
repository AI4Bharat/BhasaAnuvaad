# BhasaAnuvaad: A Speech Translation Dataset for 13 Indian Languages


[![ArXiv](https://img.shields.io/badge/arXiv-2411.02538-b31b1b.svg)](http://arxiv.org/abs/2411.04699)     [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/collections/ai4bharat/bhasaanuvaad-672b3790b6470eab68b1cb87) [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

BhasaAnuvaad, is the largest Indic-language AST dataset spanning over 44,400 hours of speech and 17M text segments for 13 of 22 scheduled Indian languages and English.

This repository contains code for the pipeline used to generate the final dataset in the NeMo format. It uses the [NeMo Forced Aligner](https://github.com/AI4Bharat/NeMo.git) to align sentences with their audio chunks, and then performs bi-text mining by using the [Sonar Text Encoder](https://github.com/facebookresearch/SONAR) to generate embeddings along with the calculation of Cosine Similarity scores.

## Usage

##### Prerequisites
- Conda Environment with Python 3.10 installed
- Support for CUDA 12.1

1. Clone this repository and setup environment
```bash
git clone https://github.com/AI4Bharat/BhasaAnuvaad.git
cd BhasaAnuvaad
bash setup.sh
```

2. Set all the values in the config.yaml as specified in the `sample_pipeline_config.yaml` file and generate the input manifest in the format specified in `sample_input_manifest.jsonl`. When going from X -> Y, one config file and input manifest will be required for each X.

3. Run pipeline
```bash
python3 main.py -c config.yaml
```

## Citation

If you use BhasaAnuvaad in your work, please cite us:

```bibtex
@article{jain2024bhasaanuvaad,
  title   = {BhasaAnuvaad: A Speech Translation Dataset for 14 Indian Languages},
  author  = {Sparsh Jain and Ashwin Sankar and Devilal Choudhary and Dhairya Suman and Nikhil Narasimhan and Mohammed Safi Ur Rahman Khan and Anoop Kunchukuttan and Mitesh M Khapra and Raj Dabre},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2411.04699}
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

For any questions or feedback, please contact:
- Raj Dabre (raj.dabre@cse.iitm.ac.in)
- Sparsh Jain (sjshiva8287@gmail.com)
- Ashwin Sankar (ashwinsankar@ai4bharat.org)
- Nikhil Narasimhan (nikhilnarasimhan@ai4bharat.org)
- Mohammed Safi Ur Rahman Khan (safikhan2000@gmail.com)

## Links

- [GitHub Repository 💻](https://github.com/AI4Bharat/BhasaAnuvaad.git)
- [Paper 📄](http://arxiv.org/abs/2411.04699)
- [Hugging Face Dataset 🤗](https://huggingface.co/collections/ai4bharat/bhasaanuvaad-672b3790b6470eab68b1cb87)


