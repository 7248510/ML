import os
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
#You need an Azure key to use the search_images_bing function
#The specific service is the Azure Cognitive Services Search API
key = os.environ.get('AZURE_SEARCH_KEY', '') # REMOVE KEY
#<function fastbook.search_images_bing(key, term, min_sz=128, max_images=150)>

#Download does not use Azure
#download_url("http://10.0.1.4/photo.png", "") #Download url will download from an image
#download_images
#<function fastai.vision.utils.download_images(dest, url_file=None, urls=None, max_pics=1000, n_workers=8, timeout=4, preserve_filename=False)>


def gatherImages():
    vehicle_types = 'cars','motorcycles'
    path = Path('vehicles2')
    if not path.exists():
        path.mkdir()
        for o in vehicle_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_bing(key, f'{o} ')
            download_images(dest, urls=results.attrgot('contentUrl'))

def validate():
    path = Path('vehicles2')
    fns = get_image_files(path)
    #print(fns) #Out put the newly created files
    failed = verify_images(fns)
    failed.map(Path.unlink)
    #print(failed)

def main():
    path = Path('vehicles2')
    vehicles = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
    dls = vehicles.dataloaders(path,number_workers=0)
    #dls.valid.show_batch(max_n=4, nrows=1)
    vehicles = vehicles.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
    dls = vehicles.dataloaders(path,num_workers=0)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)
    vehicles = vehicles.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
    dls = vehicles.dataloaders(path,num_workers=0)
    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    interp = ClassificationInterpretation.from_learner(learn)
    test = interp.plot_confusion_matrix()
    #cleaner = ImageClassifierCleaner(learn)
    learn.export()
    path = Path()
    path.ls(file_exts='.pkl')

def testing():
    #Test the model with cars V. motorcycles
    path = Path('')
    learn_inf = load_learner(path/'carvmotor.pkl')
    print(learn_inf.predict('civic.jpg')) 
    print(learn_inf.predict('Honda-Rebel.jpg'))
    print(learn_inf.predict('m2.jpg'))
    print(learn_inf.predict('AE-86.jpg'))

if __name__ == '__main__':
    #gatherImages()
    #validate()
    #main()
    testing()
