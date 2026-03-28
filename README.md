# Representing LLMs in Prompt Semantic Task Space

Presented at **Findings of the Association for Computational Linguistics: EMNLP 2025**

**Authors**: Idan Kashani, Avi Mendelson, Yaniv Nemcovsky

[Paper](https://aclanthology.org/2025.findings-emnlp.456/)

[Poster](https://assets.underline.io/lecture/132335/poster_document/eb230259203a96ebbe7f656625c288c0.pdf)

In this repository, we provide the implementation of the LLM Representation method presented in the paper.

### Abstract:
Large language models (LLMs) achieve impressive results over various tasks, and ever-expanding public repositories contain an abundance of pre-trained models. Therefore, identifying the best-performing LLM for a given task is a significant challenge. Previous works have suggested learning LLM representations to address this. However, these approaches present limited scalability and require costly retraining to encompass additional models and datasets. Moreover, the produced representation utilizes distinct spaces that cannot be easily interpreted. This work presents an efficient, training-free approach to representing LLMs as linear operators within the prompts' semantic task space, thus providing a highly interpretable representation of the models' application. Our method utilizes closed-form computation of geometrical properties and ensures exceptional scalability and real-time adaptability to dynamically expanding repositories. We demonstrate our approach on success prediction and model selection tasks, achieving competitive or state-of-the-art results with notable performance in out-of-sample scenarios.

### Reference

```
@inproceedings{kashani-etal-2025-representing,
    title = "Representing {LLM}s in Prompt Semantic Task Space",
    author = "Kashani, Idan  and
      Mendelson, Avi  and
      Nemcovsky, Yaniv",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.456/",
    doi = "10.18653/v1/2025.findings-emnlp.456",
    pages = "8578--8597",
    ISBN = "979-8-89176-335-7"
}
```
