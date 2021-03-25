This baseline method is based off of the one found in [this](https://realpython.com/python-keras-text-classification/) tutorial.

Notes:
- this uses the pre-trained glove.6B.50d model, which is stored locally to run. The current file path for this is "glove/glove.6B.50d.txt" which may need to be adjusted/downloaded to run on a different machine.
- this model is not currently set up  to run on GPUs. (Mainly due to difficulties working with the cuda library versions on ecs machines)
