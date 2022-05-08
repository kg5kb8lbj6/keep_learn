from re import M
import torch
import cv2
from cnn import Net
from torch.autograd import Variable
from torchvision import transforms

def to_onnx(model, c, w, h, onnx_name):
    dummy_input = torch.randn(1, c, w, h, device='cpu')
    torch.onnx.export(model, dummy_input, onnx_name, verbose=True)

use_cuda = False
model = Net()
model.load_state_dict(torch.load(r'D:\learn\mnist_model\params_30.pth'))

model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

to_onnx(model, 3, 28, 28,(r'D:\learn\mnist_model\params.onnx'))

img = cv2.imread(r'D:\learn\mnist_model\\4_00440.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
print('####prediction#####', prediction)
print('##torch.max(prediction, 1)###', torch.max(prediction, 1))
pred = torch.max(prediction, 1)[1]
print('pred == ',pred)
cv2.imshow("image", img)
cv2.waitKey(0)