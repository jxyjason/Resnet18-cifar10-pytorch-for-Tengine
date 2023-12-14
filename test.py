import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import torch.onnx as onnx
from PIL import Image
from torchvision import transforms



# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 100
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()  # 验证模型

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化图像
])

# 2. 加载图像
image_path = 'dataset/bird.jpg'
image = Image.open(image_path)
# 3. 应用图像预处理转换
input_tensor = transform(image).unsqueeze(0)  # 添加一个维度，将单个图像转换为批次
print(input_tensor)
# 4. 将输入张量移动到指定设备上
input_tensor = input_tensor.to(device)
with torch.no_grad():
    output = model(input_tensor)
    print(output)


# 导出onnx
# dummy_input = torch.randn(1,3,32,32)
# dummy_input = dummy_input.to(device)
# onnx_path = "resnet18_cifar10.onnx"
# onnx.export(model, dummy_input, onnx_path, verbose=True)



