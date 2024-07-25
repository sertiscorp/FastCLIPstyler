import torch
import torch.nn as nn

class TextEmbedder(nn.Module):
    def __init__(self, text_encoder):
        super(TextEmbedder, self).__init__()

        self.text_encoder = text_encoder

        if text_encoder == 'fastclipstyler':

            self.model = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 150),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(150, 150),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(150, 100),
                nn.Tanh(),
            )

        elif text_encoder == 'edgeclipstyler':

            self.model = nn.Sequential(
                nn.Linear(768, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 150),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(150, 150),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(150, 100),
                nn.Tanh(),
            )

        else:
            raise Exception("Invalid text encoder. Must be either fastclipstyler or edgeclipstyler")

    def load_model(self):
        if self.text_encoder == 'fastclipstyler':
            self.load_state_dict(torch.load("styleaug/checkpoints/text_embedder_clip.pth", map_location=torch.device('cpu')), strict=True)
        elif self.text_encoder == 'edgeclipstyler':
            self.load_state_dict(torch.load("styleaug/checkpoints/text_embedder_bert.pth", map_location=torch.device('cpu')), strict=True)

    def forward(self, text_embeddings):
        latent_space_sample = self.model(text_embeddings)
        return latent_space_sample