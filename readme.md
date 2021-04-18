This baseline method is based off of the one found in [this](https://realpython.com/python-keras-text-classification/) tutorial.

```python cnn_base.py``` in the command line to run the program.

Notes:
- this uses the pre-trained glove.6B.50d model, which is stored locally to run. The current file path for this is "glove/glove.6B.50d.txt" which may need to be adjusted/downloaded to run on a different machine.
- currently has the accuracy plot and model summary details displaying. This can be disabled by commenting out ```plt.show()``` and ```model.summary()``` respectively.
- this also uses the datasets from the same tutorial, which have been included in the ```data/``` folder. This means the model is running three times; one for each dataset.
- this model is not currently set up  to run on GPUs. (Mainly due to difficulties working with the cuda library versions on ecs machines)


TODO:
- investigate emoji handling fictionary from [this](https://studymachinelearning.com/text-preprocessing-handle-emoji-emoticon/) website.
