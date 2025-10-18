import torch
import torch.nn as nn
import torch.optim as optim
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils import Vocabulary, get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 256
hidden_size = 512
num_epochs = 5
batch_size = 4
vocab = Vocabulary()
vocab.add_word("a"); vocab.add_word("cat"); vocab.add_word("sits"); vocab.add_word("on"); vocab.add_word("mat")

data_loader = get_loader('data/sample', 'data/sample/captions.json', vocab, batch_size)

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab.word2idx)).to(device)

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=0.001)

for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(data_loader):
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}")

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'vocab': vocab.word2idx
}, 'model.pth')

print("✅ Training complete & model saved.")
