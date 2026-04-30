## Understanding the Foundations of Large Language Models via Formal Language Learning
Corresponding paper:
- Bishwamittra Ghosh, Soumi Das, Till Speicher, Qinyuan Wu, Mohammad Aflah Khan, Deepak Garg, Krishna P Gummadi, and Evimaria Terzi. **Fine-tuning vs. in-context learning in large language models: A formal language learning perspective**. In Proc. ACL, 2026. URL: https://arxiv.org/abs/2604.23267

<p align="center"><img src="illustrations/ft_vs_icl.jpg" width="60%" alt="LLMs learn in two modes. Which one is more language proficient?"></p>


<!-- - Bishwamittra Ghosh, Soumi Das, Qinyuan Wu, Mohammad Aflah Khan, Krishna P Gummadi, Evimaria Terzi, and Deepak Garg. **Understanding the Interplay between Memorization and Learning in LLMs**. In Submission, 2026.

<p align="center"><img src="illustrations/memorization_measures.png" width="60%" alt="Contextual memorization disentangles between contextual learning and memorization, and connects learning-based memorization measures to privacy-based memorization measures."></p> -->


## Installation
The code is tested on Python 3.10
```
pip install -r grammar_learning/requirements310.txt
```

## Directory Structure
- grammar_learning/training: training scripts
- grammar_learning/benchmark: scripts to generate formal language benchmarks
- grammar_learning/read_output: auxiliary scripts for processing results and plotting
- grammar_learning/data: directories for formal language benchmarks