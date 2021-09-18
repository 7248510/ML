#The images are stock
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images' #Grab the model data
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42, #0.2 = use 20% of data as the validation, seed 42 = Sets the random seed to the same value everytime we run the code
    #If we retrain the model   we know the difference is due to the changes to the model. not due to having a different random validation set
    #FastAI will always show the models accuracy using only the validation set
    label_func=is_cat, item_tfms=Resize(224), num_workers=0) #num_workers is for Windows
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(10) # A complete pass through the input data = CH1 40
imageTest = SimpleNamespace(data = ['validationSet/stock1.jpg']) #Test image of a cat
img = PILImage.create(imageTest.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}")
print(f"Probability it's a cat: {probs[1].item():.2f}%") # 2 = 2 decimal places