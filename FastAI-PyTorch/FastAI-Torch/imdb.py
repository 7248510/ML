from fastbook import * #Import jupyter
from fastai.text.all import * # Import fast ai's text library
#In the FastAI video course this was tested on an AWS GPU
#Pytorch/FastAI will most likely run out of GPU
#Run it in AWS or Google Colab
#Then export the model
#dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test',bs=8, num_workers=0) #remove num workers if != windows 
dls = TextDataLoaders.from_folder(untar_data('C:\\Users\\Caleb\\.fastai\\data\\imdb'), valid='test',bs=8, num_workers=0) #This is a local path to the dataset ^ will download the path
#num_workers = 0 == Windows
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
#learn.export() #Export the file
learn.predict("That was a really good movie!") #Run a prediction/test the model
