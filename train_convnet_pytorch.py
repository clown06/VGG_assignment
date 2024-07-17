"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Introduce related packages
import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# Selection of device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''class DataSetAdapter(Dataset):
  def __init__(self, data_set):
    self.data_set = data_set

  def __len__(self):
    # 返回数据集中的样本总数
    return self.data_set.num_examples

  def __getitem__(self, index):
    # 获取一个批次的数据
    images, labels = self.data_set.next_batch(batch_size=32)
    return images, labels'''

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  _, predicted_labels = torch.max(predictions, 1)
  _, true_labels = torch.max(targets, 1)
  correct_predictions = (predicted_labels == true_labels).sum().item()
  accuracy = correct_predictions / targets.size(0)
  return accuracy

  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  torch.manual_seed(42)

  # Load CIFAR-10 dataset
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  # cifar10_Dataset = DataSetAdapter(cifar10)
  train_loader = DataLoader(cifar10['train'], batch_size=FLAGS.batch_size, shuffle=True)
  test_loader = DataLoader(cifar10['test'], batch_size=FLAGS.batch_size, shuffle=False)

  # Initialize the model, loss function, and optimizer
  model = ConvNet(3, 10)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  # Training loop
  train_losses = []
  train_accuracies = []
  test_accuracies = []
  step = 0

  while step < FLAGS.max_steps:
    model.train()
    model.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      data = data.to(device)
      # target = target.long().to(device)
      target = target.to(device)
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())
      train_acc = accuracy(output, target)
      train_accuracies.append(train_acc)

      # Output training loss and accuracy for every 100 steps
      if (step) % 100 == 0 or step == FLAGS.max_steps - 1:
        print(f'Train Step: {step} Loss: {loss.item()} Accuracy: {train_acc}')


      step += 1
      # Test every 500 steps on the test set
      if step % 500 == 0:
        model.eval()
        test_correct = 0
        test_total = 0
        for data, target in test_loader:
          data = data.to(device)
          output = model(data)
          output = output.cpu()

          test_correct += (output.argmax(1) == target.argmax(1)).sum().item()
          test_total += target.size(0)

        test_accuracy = test_correct / test_total
        test_accuracies.append(test_accuracy)
        print(f'Test Step: {step} Accuracy: {test_accuracy}')
      if step >= FLAGS.max_steps:
        break


  # Plotting the training and evaluation results
  test_steps = np.arange(500, 5500, 500)
  plt.figure(figsize=(10, 8))
  # Figure 1: Training loss
  plt.subplot(2, 1, 1)
  plt.plot(train_losses, label='Train Loss', color='blue')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.legend()

  # Figure 2: Training and testing accuracy
  plt.subplot(2, 1, 2)
  plt.plot(train_accuracies, label='Train Accuracy', color='green')
  plt.plot(test_steps, test_accuracies, label='Test Accuracy', color='red', linestyle='--')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  plt.title('Training and Test Accuracy')
  plt.legend()

  plt.tight_layout()
  plt.show()

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()