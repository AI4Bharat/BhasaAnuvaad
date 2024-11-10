# BhasaAnuvaad: A Speech Translation Dataset for 13 Indian Languages


[![ArXiv](https://img.shields.io/badge/arXiv-2411.02538-b31b1b.svg)](https://arxiv.org/abs/2411.02538)     [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/collections/ai4bharat/bhasaanuvaad-672b3790b6470eab68b1cb87) [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

BhasaAnuvaad, is the largest Indic-language AST dataset spanning over 44,400 hours of speech and 17M text segments for 13 of 22 scheduled Indian languages and English.

This repository contains code for the pipeline used to generate the final dataset in the NeMo format. It uses the [NeMo Forced Aligner](https://github.com/AI4Bharat/NeMo.git) to align sentences with their audio chunks, and then performs bi-text mining by using the [Sonar Text Encoder](https://github.com/facebookresearch/SONAR) to generate embeddings followed by the calculation of Cosine Similarity scores.
