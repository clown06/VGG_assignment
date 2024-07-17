# Introduction
In this version, the VGG model was mainly modified and built for **train_comvnet_pytorch. py** and **convnet_pytorch. py**, and training and testing were completed. In addition, for the convenience of using DataLoader to load data, slight modifications were made to **cifar10_utils. py**, and comments were added (the __len__ and __getitem__ functions were added to the Dataset class).

If the code in **cifar10_utils. py** is not modified, the following code can be added to **train_comvnet_pytorch. py**.
And after loading the CIFAR-10 dataset, call it in the following way.

cifar10_Dataset = DataSetAdapter(cifar10)

```python
class DataSetAdapter(Dataset):
  def __init__(self, data_set):
    self.data_set = data_set

  def __len__(self):
    # Return the total number of samples in the dataset
    return self.data_set.num_examples

  def __getitem__(self, index):
    # Obtain data for a batch
    images, labels = self.data_set.next_batch(batch_size=32)
    return images, labels
```

# Experimental Result
The accuracy and loss curves drawn are shown in reault.png.
The train_losses, train_accuracies, and test_accuracies in the experiment are shown in the results. pdf.(Print train_losses and train_accuracies every 100 steps, and test them on the test set every 500 steps)
