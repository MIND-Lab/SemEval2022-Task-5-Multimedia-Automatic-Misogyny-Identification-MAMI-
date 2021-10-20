# SemEval2022-Task 5: Multimedia Automatic Misogyny Identification (MAMI)
This is the Github repository for SemEval-2022 Task 5, **Multimedia Automatic Misogyny Identification** (**MAMI**). This repositopry contains:
1. Dataset Request Form
2. Evaluation Script
3. Baselines
4. Contacts

## Dataset Request Form
The datasets are exclusively reserved for the participants of SemEval-2022 Task 5 and are not to be used freely. They may be distributed upon request and for academic purposes only. To request the datasets, please fill in the following form: https://forms.gle/AGWMiGicBHiQx4q98

After submitting the required info, participants will have  a link to a folder containing the datasets in a zip format (trial, training and development) and the password to uncompress the files.

## Evaluation Script
The evalulation script are used to rank the teams participating in the MAMI challenge, estimating macro-average F1-Measure for Subtask A and weighted-average F1-Measure for Subtask B. The evaluation script is available in the [Evaluation Folder](https://github.com/MIND-Lab/MAMI/tree/main/Evaluation).


## Baselines
The baseline models used for the MAMI challenge can be found in [Baseline Folder](https://github.com/MIND-Lab/MAMI/tree/main/Baselines).
For Subtask A, the baselines are grounded on:
1. a deep representation of text, i.e. a fine-tuned sentence embedding using the USE pre-trained model;
2. a deep representation of image content, i.e. based on a fine-tuned image classification model grounded on VGG-16;
3. a concatenation of deep image and text representations, i.e. a single layer neural network.

For Subtask B, the baselines are grounded on:
1. a multi-label model, based on the concatenation of deep image and text representations, for predicting simpultaneosly if a meme is misogynous and the corresponding type;
2. a hierarchical multi-label model, based on the concatenation of deep image and text representations, for predicting if a meme is misogynous or not and, if misogynopus, the corresponding type.

## Contacts
Should you have any questions, please join the MAMI Google group: semeval2022-mami_AT_googlegroups.com.
