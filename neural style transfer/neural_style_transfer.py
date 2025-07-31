import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image loader with small size
def image_loader(image_path, max_size=128, shape=None):
    loader = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Show image
def imshow(tensor, title=None):
    image = tensor.detach().cpu().clone().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Load images (change filenames if needed)
content_img = image_loader("content.jpg")
style_img = image_loader("style.jpg", shape=[content_img.size(2), content_img.size(3)])
input_img = content_img.clone()

assert content_img.size() == style_img.size(), "Images must be the same size!"

# VGG normalization
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalization layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Loss layers
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Build style transfer model
def get_style_model(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Style transfer runner
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                       num_steps=50, style_weight=1e5, content_weight=1):
    model, style_losses, content_losses = get_style_model(cnn, norm_mean, norm_std, style_img, content_img)
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)

    print("Starting Style Transfer...\n")
    for step in range(num_steps):
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

    input_img.data.clamp_(0, 1)
    return input_img

# Run it
output = run_style_transfer(cnn, cnn_mean, cnn_std,
                            content_img, style_img, input_img)

# Show final result
imshow(output, title="Stylized Output")
