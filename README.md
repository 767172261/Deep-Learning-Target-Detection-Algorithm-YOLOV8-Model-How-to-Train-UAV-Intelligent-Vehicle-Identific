深度学习目标检测算法YOLOV8模型如何训练 无人机智能车辆识别计数数据集 无人机车辆计数检测数据集 车辆检测计数数据集的训练及应用


**

## 数据集描述：

**
车辆检测计数数据集，可用于车辆计数【CARPK】
train:1158张
val:290张
一类别car
可直接可以用于目标检测算法训练。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ada2863839e4525a59a2e2463eb274a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7eb3ced5544f4965ae71ed89b0bd04f9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a307e609ee24672bfcafe345058d621.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3989fcd841543eda77d43e0f26be62f.png)

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/222cabaf3978409bb728d89765f94991.png)



1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae93f8bd0c3d42e88c6f3e95319bd3cb.png)


---

## ✅ 一、数据集概览

| **项目** | **详情** |
|--------|--------|
| 数据集名称 | CARPK（Car Counting and Detection Dataset） |
| 任务类型 | 单类别目标检测（车辆检测 + 计数） |
| 图像总数 | 1448 张（train: 1158, val: 290） |
| 类别数量 | 1 类：`car` |
| 图像特点 | 高空视角（无人机/摄像头拍摄），车辆尺寸小、密度高 |
| 适用场景 | 城市交通监控、停车场车辆计数、智能交通系统（ITS） |
| 标注格式 | YOLO 格式（.txt），已划分 train/val |
| `data.yaml` | 已配置为相对路径，可直接加载 |

---

## ✅ 二、目录结构建议

```
CARPK/
├── images/
│   ├── train/          # 1158 张训练图像
│   └── val/            # 290 张验证图像
├── labels/
│   ├── train/          # 对应的 YOLO .txt 标注文件
│   └── val/
├── data.yaml           # 数据配置文件（相对路径）
└── README.md
```

### `data.yaml` 示例（确认内容如下）

```yaml
train: ./images/train
val: ./images/val

nc: 1
names: ['car']
```

---

## ✅ 三、YOLOv8 训练代码

### `train.py`

```python
from ultralytics import YOLO
import torch

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载预训练模型（推荐使用小模型，因车辆小且密集）
model = YOLO("yolov8s.pt")  # 可选: yolov8n, yolov8m（平衡速度与精度）

# 开始训练
results = model.train(
    data='data.yaml',           # 相对路径，自动识别
    epochs=150,                 # 训练轮数
    imgsz=1024,                 # 高分辨率输入（CARPK 图像通常较大）
    batch=16,                   # 根据显存调整（16-32）
    optimizer='AdamW',
    lr0=0.001,                  # 初始学习率
    lrf=0.1,                    # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    patience=20,                # 早停：20轮无提升则停止
    project="runs/CARPK",
    name="exp_yolov8s_1024",    # 实验名称
    save=True,
    save_period=10,
    device=device,
    workers=8,                  # 数据加载线程数
    augment=True,               # 启用数据增强
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=15,            # 最后15轮关闭 Mosaic（提升收敛）
)
```

---

## ✅ 四、模型验证

```python
# 验证模型性能
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

> 💡 **提示**：CARPK 数据集常用于计数任务，可通过 `Recall` 指标评估漏检率。

---

## ✅ 五、推理与车辆计数代码

### 1. 单图推理 + 车辆计数

```python
# predict_and_count.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("runs/CARPK/exp_yolov8s_1024/weights/best.pt")

# 推理单张图像
img_path = "test_car_park.jpg"
results = model.predict(
    source=img_path,
    imgsz=1024,
    conf=0.3,
    iou=0.5,
    save=False,                  # 不保存，自定义绘制
    show_labels=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 获取结果
for r in results:
    orig_img = r.orig_img  # 原始图像 (H, W, C)
    boxes = r.boxes.xyxy.cpu().numpy()  # 边界框
    confs = r.boxes.conf.cpu().numpy()  # 置信度
    cls_ids = r.boxes.cls.cpu().numpy()  # 类别 ID

    # 绘制边界框和计数
    count = len(boxes)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f'{confs[i]:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示车辆总数
    cv2.putText(orig_img, f"Total Cars: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 保存结果
    output_path = "output_counted.jpg"
    cv2.imwrite(output_path, orig_img)
    print(f"检测完成，共 {count} 辆车，结果保存至: {output_path}")
```

---

### 2. 批量推理（整个文件夹）

```python
# 批量处理
results = model.predict(
    source="CARPK/images/val",      # 验证集图像目录
    save=True,
    project="inference",
    name="CARPK_results",
    imgsz=1024,
    conf=0.3,
    device=device
)

# 统计总检测数量
total_cars = 0
for r in results:
    total_cars += len(r.boxes)
print(f"在 {len(results)} 张图像中，共检测到 {total_cars} 辆车。")
```

---

## ✅ 六、车辆计数优化建议

| 技术 | 说明 |
|------|------|
| **滑动窗口** | 对超大图像分块检测，避免小车漏检 |
| **后处理 NMS 调优** | 降低 `iou=0.3~0.4` 防止密集车辆被合并 |
| **TTA（Test Time Augmentation）** | 提升小目标召回率 |
| **使用 YOLOv8n + 高分辨率** | 在边缘设备部署时兼顾速度与精度 |

---

## ✅ 七、导出模型（ONNX/TensorRT）如果有需要的话

```python
# 导出为 ONNX（用于部署）
model.export(format="onnx", dynamic=True, opset=12, imgsz=1024)

# 导出为 TensorRT（高性能推理）
model.export(format="engine", half=True, dynamic=True, imgsz=1024)
```

---

## ✅ 总结

- ✅ **数据集准备完成**：`data.yaml` 已为相对路径，可直接训练
- ✅ **训练代码完整**：支持 YOLOv8s/m/l/x，适配高分辨率图像
- ✅ **推理 + 计数一体化**：输出检测框并自动统计车辆数量
- ✅ **适用于**：智能停车、交通流量分析、无人机巡检等场景


