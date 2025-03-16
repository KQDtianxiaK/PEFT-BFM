## PEFT-BFM
（Parameter-Efficient Fine-Tuning For Biological Foundation Models)
# File description
**Checkpoints:** Model file saved after fine-tuning

**Datasets:** Datasets for downstream tasks fine-tuned in this study

**Model:** Models tested and used in this study (model weights file not uploaded)

**Metrics:** Evaluation metrics file used in this study

**Scripts:** Script file for fine-tuning the model
# Research motivation
The current biological basic models face the problems of high computing power requirements and data annotation costs. This study aims to introduce the PEFT method into the field of biological basic models. By only fine-tuning the model parameters that basically do not exceed 3%, while reducing the consumption of computing resources, verify its performance in DNA/RNA/protein downstream tasks, and provide a new paradigm for the lightweight application of large biological models.
# Research content
Based on the model size, training data, and open source level, biological basic models trained with sequence data such as DNA, RNA, and Protein based on the Transformer architecture were selected. Various PEFT methods such as LoRA, OFT, and LayerNorm Tuning were used to evaluate the fine-tuning performance, training time optimization, and computing resource usage of the three types of biological models in various downstream tasks using different PEFT methods. The parameter efficient fine-tuning method selects and fine-tunes some parameters of the model, adds prompts to the input or inside the model, adds additional parameters or modules to the model, and freezes the original parameters of the model. While maintaining its representation learning ability, it continues to train a very small number of parameters to improve the performance of the model on downstream tasks while greatly reducing the demand for computing resources, providing strong support for the application of large models and research in related fields.