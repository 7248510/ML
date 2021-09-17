import torch
print("Make sure you install the correct pytorch. CPU is the default choice.\nTo install Cuda 11.2 drivers go to Nvidia's website.")
x = torch.rand(5, 3) #Initialized a random tensor
print(x) # Print the results
print("Cuda available: ",torch.cuda.is_available())
print("Current Cuda Device: ",torch.cuda.current_device())