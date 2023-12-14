import onnxruntime
import numpy as np
from PIL import Image

# 加载模型
model_path = "resnet18_cifar10.onnx"
session = onnxruntime.InferenceSession(model_path)

# 加载图片并进行预处理
image_path = "dataset/bird.jpg"
image = Image.open(image_path)
# 在这里可以进行图像预处理，如缩放、裁剪、归一化等
# 假设输入的图像尺寸为 (224, 224) 并且通道顺序为 RGB
image = np.array(image).astype(np.float32)
image = np.divide(image, 255.0)  # 归一化到 [0.0, 1.0]
image = np.transpose(image, (2, 0, 1))  # 调整通道顺序
image = np.expand_dims(image, axis=0)  # 添加批次维度

# 执行推理
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: image})

# 处理输出
output = outputs[0]
# 在这里可以对输出进行后处理，如解码分类结果、绘制边界框等
# 这里假设输出是一个概率向量，你可以根据需求进行处理

# 打印结果
print(output)