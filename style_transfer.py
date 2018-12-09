import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, models

###############################################################################
############## 1. LOAD Pre-Trained VGG19 Model & FREEZE WEIGHTS ###############
###############################################################################
vgg = models.vgg19(pretrained = True).features

for param in vgg.parameters():
    param.requires_grad = False   # freeze weights
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)



###############################################################################
####################### 2. LOAD CONTENT & STYLE IMAGES ########################
###############################################################################
def load_image(img_path, max_size = 400, shape = None):
    # load  & transform an image to be <= 400 px in width & height (large img: slow to process)
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
        
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    # discard transparent, alpha channel && add batch dimension:
    image = transform(image)[:3, :, :].unsqueeze(0)
    
    return image


# match style image's size same as content image
content_image = load_image('images/washed_out.jpg').to(device)
style_image = load_image('images/moasic.jpg', shape = content_image.shape[-2:]).to(device)



###############################################################################
################### 3. VISUALIZE CONTENT & STYLE IMAGES #######################
###############################################################################
# un-normalizing & converting image from a Tensor image to a NumPy image for display
def im_convert(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1) # maintain values within [0,1] range

    return image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
ax1.imshow(im_convert(content_image))
ax2.imshow(im_convert(style_image))



###############################################################################
######################## 4. PASS IMAGES THROUGH VGG ###########################
######## && Extract 6 Required Layers as a DICT {Layer Name: Weights} #########
###############################################################################
# only extract 6 required feature map layers from vgg19
def get_features(image, model, layers = None):
    # mapping layer names of PyTorch's VGGNet to names from the paper
    if layers is None:
        six_layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',
                      '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)  # Applying vgg's frozen weights to the image pixel values
        if name in six_layers:
            features[six_layers[name]] = x
            
    return features

# Dict w/ 6 elements: keys- conv#_#// values- weights of that layer
content_features = get_features(content_image, vgg)  
style_features = get_features(style_image, vgg)



###############################################################################
################## 5. COMPUTE GRAM MATRIX FOR STYLE IMAGE #####################
###############################################################################
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram

style_gram_matrices = {layer: gram_matrix(style_features[layer]) for layer in style_features}



###############################################################################
##################### 6. INITIALIZE RESULT 'TARGET' IMAGE #####################
###############################################################################
# result image: 'target' --> prep if for change (good start is the copy of content image)
target_image = content_image.clone().requires_grad_(True).to(device)



###############################################################################
############### 7. SET STYLE LAYER WEIGHTS FOR LOSS CALCULATION ###############
###############################################################################
# How important each layer when calculating Style Loss
# weighting earlier layers more will result in LARGER STYLE artifacts
style_loss_weights = {'conv1_1': 1., 
                      'conv2_1': 0.4,
                      'conv3_1': 0.2, 
                      'conv4_1': 0.1, 
                      'conv5_1': 0.05}

content_alpha = 1
style_beta = 1e4



###############################################################################
############################# 8. TRAIN THE NETWORK ############################ 
############################################################################### 
criterion = nn.MSELoss()
optimizer = optim.Adam([target_image], lr = 0.003)
print_every = 500
n_epochs = 3000


for epoch in range(n_epochs):
    target_features = get_features(target_image, vgg)
    
    content_loss = criterion(target_features['conv4_2'], content_features['conv4_2']) 
    
    style_loss = 0
    for layer in style_loss_weights:                                          
        #style_loss += style_weights[layer] * criterion(gram_matrix(target_features[layer]), gram_matrix(style_features[layer]))
        style_loss_this_layer = style_loss_weights[layer] * criterion(gram_matrix(target_features[layer]), style_gram_matrices[layer])
        _, d, h, w = target_features[layer].size()
        style_loss += style_loss_this_layer / (d*h*w)
   
    total_loss = content_alpha * content_loss + style_beta * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()     # Target image gets updated here
    optimizer.step()
    
    
    if ((epoch+1) % print_every == 0) or (epoch==0):
        print('Epoch: {}\t'.format(epoch+1), 'Total Loss: {:.4f}'.format(total_loss.item()))
        plt.imshow(im_convert(target_image))
        plt.show()
    
       
        
###############################################################################
################## 9. VISUALIZE ORIGINAL & FINAL TARGET IMAGE #################
###############################################################################         

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(im_convert(content_image))
ax2.imshow(im_convert(target_image))

plt.imsave('final.jpg', im_convert(target_image))
        