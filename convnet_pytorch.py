"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    '''
    32/64/16/16
    32/128/8/8
    32/256/8/8
    32/256/4/4
    32/512/4/4
    32/512/2/2
    32/512/2/2
    32/512/1/1
    32/512
    32/10
    '''

    super(ConvNet, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.conv3_a = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.conv3_b = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.conv4_a = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.conv4_b = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.conv5_a = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.conv5_b = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.fc = nn.Linear(512, n_classes)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3_a(out)
    out = self.conv3_b(out)
    out = self.conv4_a(out)
    out = self.conv4_b(out)
    out = self.conv5_a(out)
    out = self.conv5_b(out)
    out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
    out = self.fc(out)
    return out

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
