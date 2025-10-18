import torch
from torchvision import transforms
from PIL import Image
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('model.pth', map_location=device)
vocab = Vocabulary()
vocab.word2idx = checkpoint['vocab']
vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}

encoder = EncoderCNN(256).to(device)
decoder = DecoderRNN(256, 512, len(vocab.word2idx)).to(device)

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval(); decoder.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    feature = encoder(image)
    ids = decoder.sample(feature)
    words = vocab.decode(ids)
    return " ".join(words)

caption = predict('data/sample/image1.jpg')
print("Generated Caption:", caption)
