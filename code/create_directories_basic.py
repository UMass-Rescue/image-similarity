import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import sys
from PIL import Image
import shutil

# the architecture to use - this can be changed, but resnet seems to give the best results
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

if not os.path.isdir(sys.argv[2]):
    os.mkdir(sys.argv[2])

directory = sys.argv[1]
for filename in os.listdir(directory):
    img = Image.open(directory + '/' + filename)
    #img = Image.open(img_name)
    try:
        input_img = V(centre_crop(img).unsqueeze(0))
    except RuntimeError:
        continue

    # forward pass
    logit = model.forward(input_img)    
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{} prediction on {}'.format(arch,filename))
    class_final = classes[idx[0]].replace('/','_',10)
    # output the prediction
    if os.path.isdir(sys.argv[2] + '/' + class_final):
        shutil.copy2(sys.argv[1] +  filename, sys.argv[2] + '/' + class_final)
    else:
        os.mkdir(sys.argv[2] + '/' + class_final)
        shutil.copy2(sys.argv[1] + filename, sys.argv[2] + '/' + class_final)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))