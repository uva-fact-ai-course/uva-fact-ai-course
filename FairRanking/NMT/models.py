import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device = 'cpu'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = 10, device = 'cpu', uniform_attention = False):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.device = device
        self.uniform_attention = uniform_attention

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        if self.uniform_attention:
            attn_weights = 1 / self.max_length * torch.ones(1, self.max_length)
        else:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_weights = attn_weights.to(self.device)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


    def permutation_experiment(self, input, hidden, encoder_outputs):

        with torch.no_grad():
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)

            # does not make sense for the uniform attention to be permuted!
            assert not self.uniform_attention

            # original output of this step

            original_attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

            attn_applied = torch.bmm(original_attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))

            original_output = torch.cat((embedded[0], attn_applied[0]), 1)
            original_output = self.attn_combine(original_output).unsqueeze(0)

            original_output = F.relu(original_output)
            original_output, hidden_next = self.gru(original_output, hidden)

            original_output = F.log_softmax(self.out(original_output[0]), dim=1)

            # preallocate TVDs

            TVD = torch.zeros(100)

            # 100 times permute:

            for i in range(100):

                permuted_att_weights = original_attn_weights[0, torch.randperm(self.max_length)].view(1, -1)
                # now compute "new" output
                attn_applied = torch.bmm(permuted_att_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))

                permuted_output = torch.cat((embedded[0], attn_applied[0]), 1)
                permuted_output = self.attn_combine(permuted_output).unsqueeze(0)

                permuted_output = F.relu(permuted_output)
                permuted_output, _ = self.gru(permuted_output, hidden)

                permuted_output = F.log_softmax(self.out(permuted_output[0]), dim=1)

                # TVD. Notice exp because it is the LOG softmax.

                TVD[i] = 0.5 * (abs(permuted_output.exp() - original_output.exp())).sum()

            # as the original paper did
            median_error = TVD.median()

        return original_output, hidden_next, original_attn_weights, median_error.item(), original_attn_weights.max()

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
