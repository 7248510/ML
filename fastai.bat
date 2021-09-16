echo "This assumes conda is in your path(Reinstalling Conda is not a problem as long as your know how paths/sys vars work)"
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c nvidia -c fastai fastai anaconda
echo "These commands will set up pytorch and Fast AI on a local computer with Cuda."
