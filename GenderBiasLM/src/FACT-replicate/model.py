import torch
import torch.nn as nn
from var_dropout import VarDropout


class LSTM(nn.Module):
    """This is the AWD-LSTM based on the paper:

    Regularizing and Optimizing LSTM Language Models by Merity et al (2018)

    It incorporates numerous regularization techniques such as dropout.
    """
    def __init__(self, vocab_size, device, embedding_size=400,
                 hidden_size=1150, lstm_num_layers=3, batch_size=40,
                 dropoute=0.1, dropouti=0.65, dropouth=0.3, dropouto=0.4,
                 weight_drop=0.5, tie_weights=True):
        super(LSTM, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_num_layers = lstm_num_layers
        self.weight_drop = weight_drop
        self.tie_weights = tie_weights

        self.dropoute = dropoute  # Embedding (input to embedding)
        self.dropouti = dropouti  # Input (embedding to lstm)
        self.dropouth = dropouth  # Hidden (hidden layers of lstm)
        self.dropouto = dropouto  # Output (lstm to decoder)

        self.vardropout = VarDropout().to(device)
        self.embed = nn.Embedding(vocab_size, embedding_size).to(device)
        self.layers = [nn.LSTM(embedding_size if l == 0 else hidden_size,
                               embedding_size if l == lstm_num_layers - 1 and tie_weights else hidden_size,
                               1, batch_first=True) for l in range(lstm_num_layers)]
        self.LSTM = nn.ModuleList(self.layers).to(device)

        self.decoder = nn.Linear(hidden_size, vocab_size).to(device)

        if tie_weights:
            self.decoder.weight = self.embed.weight

        self.initialize_hidden()

    def initialize_hidden(self):
        self.hidden = [(nn.Parameter(torch.zeros(1, self.batch_size,
                                     self.embedding_size if l == self.lstm_num_layers - 1 and self.tie_weights else self.hidden_size,
                                     device=self.device)),
                        nn.Parameter(torch.zeros(1, self.batch_size,
                                     self.embedding_size if l == self.lstm_num_layers - 1 and self.tie_weights else self.hidden_size,
                                     device=self.device)))
                       for l in range(self.lstm_num_layers)]

    def detach_hidden(self):
        def repackage_hidden(h):
            """Wraps hidden states in new Tensors,
            to detach them from their history."""
            if isinstance(h, torch.Tensor):
                return h.detach()
            else:
                return tuple(repackage_hidden(v) for v in h)

        self.hidden = [h for h in repackage_hidden(self.hidden)]

    def embed_dropout(self, x):
        """Do dropout on word embeddings."""
        if self.dropoute and self.training:
            mask = torch.zeros(self.embed.weight.size(0), 1, device=self.device)
            mask = mask.bernoulli_(1 - self.dropoute).expand_as(self.embed.weight)
            mask = mask / (1 - self.dropoute)
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        return nn.functional.embedding(x, masked_embed_weight, padding_idx,
                                       self.embed.max_norm,
                                       self.embed.norm_type,
                                       self.embed.scale_grad_by_freq,
                                       self.embed.sparse)

    def set_weights(self, lstm):
        """Does weight drop for the final lstm layer."""
        if self.weight_drop and self.training:
            weight_name = 'weight_hh_l0' # Hardcoded, but should find proper weights in lstm (final layer)
            weight = lstm._parameters[weight_name].detach()
            weight = nn.functional.dropout(weight, p=self.weight_drop,
                                           training=self.training)
            lstm._parameters[weight_name] = weight
            lstm.flatten_parameters()

    def forward(self, x, return_out=False):
        embed = self.embed_dropout(x.T)
        embed = self.vardropout(embed, self.dropouti)
        output = embed

        for l, lstm in enumerate(self.LSTM):
            self.set_weights(lstm)
            output, hidden = lstm(output, self.hidden.pop(0))
            self.hidden.append(hidden)
            if l != self.lstm_num_layers - 1:
                output = self.vardropout(output, self.dropouth)

        drop_output = self.vardropout(output, self.dropouto)

        if return_out:
            return self.decoder(drop_output), self.hidden, output, drop_output

        return self.decoder(drop_output), self.hidden
