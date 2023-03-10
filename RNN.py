import torch
import torch.nn as nn

class RecurrentClassifier(nn.Module):
  def __init__(self, num_layers, hidden_size=64, model_type = 'rnn'):
    super().__init__()
    
    self.output_size = 1
    dropout = 0.0
    if model_type == 'rnn':
      self.model = nn.RNN(input_size = 1, 
                          hidden_size = hidden_size,
                          num_layers = num_layers, 
                          nonlinearity = 'tanh', dropout = dropout,  batch_first = True)
    else:
      self.model = nn.LSTM(input_size = 1, 
                          hidden_size = hidden_size,
                          num_layers = num_layers, dropout = dropout, batch_first = True)

    self.linear = nn.Linear(hidden_size, self.output_size)

  def forward(self, x: torch.Tensor, h0 = None) -> torch.Tensor:

    # output features (h_t) from the last layer of the RNN
    # h_n the final hidden state for each element in the batch
    output, h_n = self.model(x, h0)
    output = self.linear(output)
    
    return output
