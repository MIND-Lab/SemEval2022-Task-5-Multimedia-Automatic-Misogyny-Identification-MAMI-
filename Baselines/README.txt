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
