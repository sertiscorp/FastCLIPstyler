import numpy as np
import torch
import torch.nn

from styleaug.ghiasi import Ghiasi
from styleaug.pbn_embedding import PBNEmbedding
from styleaug.text_embedder import TextEmbedder
from sentence_transformers import SentenceTransformer
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastCLIPStyler:

    def __init__(self, opt):

        self.opt = opt

        assert (opt.img_width % 8) == 0, "width must be multiple of 8"
        assert (opt.img_height % 8) == 0, "height must be multiple of 8"

        print('Loading Ghiasi model')
        self.styleaug = Ghiasi()
        self.styleaug.to(device)
        self.styleaug.requires_grad_(False)

        print('Loading text embedder')
        self.text_embedder = TextEmbedder(self.opt.text_encoder)
        self.text_embedder.to(device)
        self.text_embedder.requires_grad_(True)

        print('Loading PBN statistics')
        self.pbn_embedder = PBNEmbedding()
        self.pbn_embedder.to(device)
        self.pbn_embedder.requires_grad_(False)

        if opt.text_encoder == 'fastclipstyler':
            print('Loading CLIP')
            self.clip_model, _ = clip.load('ViT-B/32', device, jit=False)
            self.clip_model.to(device)
            self.clip_model.requires_grad_(False)

        elif opt.text_encoder == 'edgeclipstyler':
            print('Loading albert')
            self.bert_encoder = SentenceTransformer('paraphrase-albert-small-v2')

        else:
            raise Exception('Invalid text encoder. Should be either fastclipstyler or edgeclipstyler')

        print('Finished loading all the models')
        print()

        text_source = np.loadtxt('styleaug/checkpoints/source_array.txt')
        self.text_source = torch.Tensor(text_source).to(device)


    def _set_up_features(self):

        with torch.no_grad():

            if self.opt.text_encoder == 'fastclipstyler':

                tokens = clip.tokenize([self.opt.text]).to(device)
                clip_embeddings = self.clip_model.encode_text(tokens).detach()
                clip_embeddings = clip_embeddings.mean(axis=0, keepdim=True)
                clip_embeddings = clip_embeddings.type(torch.float32)
                clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)

                self.text_features = clip_embeddings

            elif self.opt.text_encoder == 'edgeclipstyler':

                bert_embeddings = self.bert_encoder.encode([self.opt.text])
                bert_embeddings = bert_embeddings.mean(axis=0, keepdims=True)
                bert_embeddings /= np.linalg.norm(bert_embeddings)
                bert_embeddings = torch.Tensor(bert_embeddings).to(device)

                self.text_features = bert_embeddings

    def test(self):

        self.text_embedder.load_model()
        self.text_embedder.eval()

        self._set_up_features()
        painting_embedding = self.text_embedder(self.text_features)
        target = self.styleaug(self.content_image, painting_embedding)

        return target