BASELINE
In this repository many baseline models are proposed as an example to satisfy taskA or taskB.
Each file loads the challenge data, obtaining them from a path that can be customized by modifying the first lines of the code.
Then, a specific model is created and used to make predictions on the test data. Finally, the metrics indicated by the challenge
(macro-average F1-Measure for taskA and weighted-average F1-Measure for taskB) are computed using the script evaluation.py.
Although submitting the challenge requires only a prediction file, here an example of score evaluation is provided,
in addition to the trained model and a scheme that represents its structure.

EVALUATION
As said, each baseline script evaluate the model performances using evaluation.py.
To be executed, it needs a truth.txt file used to verify the correctness of predictions. 
This file should report real labels for each predicted instance, following the below structure:

filename[tab]{0|1}{tab}{0|1}[tab]{0|1}[tab]{0|1}[tab]{0|1}

example:
1.jpg	1	1	0	0	1
2.jpg	0	0	0	0	0
3.jpg	0	0	1	0	1

Similarly, model predictions are saved in a answer.txt file by each baseline model.
To run 'evaluation.py' or any baseline script, the file 'truth.txt' must be placed in a folder name 'ref', as shown in the below 'Execution' section. 

EXECUTION
To execute all the baseline it is necessary to satisfy the requirements indicated in the relative file (requirements.txt). 
As reported in all baseline files, requirements can be installed via the command "pip install -r requirements.txt".

To run each baseline model, the following items in the same folder are required:
- evaluation script (evaluation.py, available in the folder 'Evaluation')
- train and test data as csv
- the Ground Truth file (truth.txt) in the folder 'ref'
- folder 'TRAINING' with the images provided by the challenge

input/
|_ evaluation.py
|_train.csv
|_test.csv
|_ ref/
|  |_truth.txt
|_ TRAINING/
  |_ 1.jpg
  |_ 2.jpg
  |_ ...
  
  
OUTPUT
At the end of the execution, a folder related to each specific baseline will be created. Each folder will contain:
- a dump of the model (.h5)
- a .png file showing the architecture of the model
- the score obtained from the script evalutation: scores.txt
- a 'res' folder containing the prediction for each meme (answer.txt)

output/
|_ model.h5
|_ model.png
|_ scores.txt
|_ res/
  |_answer.txt
