# To train the model from a commandline, follow the following instructions:
1. Make sure your computer installed all prequisite library such as: transformers, nlp, pyarrows, ipywidgets...
2. Run the following command from your terminal:

## To teach the model:
python3 /content/t5workcontinousimprovement.py /content/TrainingLibary /content/SavedModel/ /content/training_data.csv 5 4 KnowledgeUpdate

## To up date the knowledge libary
python3 /content/t5workcontinousimprovement.py /content/TrainingLibary /content/SavedModel/ /content/training_data.csv 5 4 KnowledgeUpgrade

## To train the model after you teach it
python3 /content/t5workcontinousimprovement.py /content/TrainingLibary /content/SavedModel/ /content/training_data.csv 5 4 Training

## To confirm what has it learned
python3 /content/t5workcontinousimprovement.py /content/TrainingLibary /content/SavedModel/ /content/training_data.csv 5 4 Confirmation

### Note:
- /content/t5workcontinousimprovement.py: is the file path where you save this python file on your computer
- /content/TrainingLibary: is a folder directory where you will save your good training examples
- /content/SavedModel/: is a folder where you will save your training model
- /content/training_data.csv: is the file path to new example you wish your computer to learn
- 5 is how often you want to see the loss print output
- 4 is the number of training epochs




