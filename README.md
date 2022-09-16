该代码库包含 AIMP 上 OCR 管道工作流所需的脚本。该管道由两个阶段组成——车牌检测和车牌识别 ocr。工作流 yaml 文件定义为 workflow.yaml。

对于车牌检测，目标检测模型是从 CVAT 训练的。默认工作流会导出冻结图，可在 license_detection.py 中使用以生成记录车牌检测框的 JSON 格式文件输出。

一旦我们得到了检测模型的输出，就可以将其输入到用于车牌识别 OCR 的 Attention OCR 模型（此示例中使用 LPRNet 预训练 OCR 模型）通过 demo_inference.py 运行推理。

The repository contains scripts required for OCR Pipeline Workflow on Onepanel. This pipeline consist of two stages-- license plate detection and license plate ocr. 

For the detection, the object detection model was trained from CVAT. The default Workflow also exports frozen graph which can be used in `license_detection.py` to generate the output. 

Once we have the output from detection model, it can fed to the Attention OCR model for OCR. Following steps demonstrate the training and inference process for this model.

### 1 - Generate TFRecords

```python3
python generate_tfrecords.py \
        --charset_path=/mnt/src/train/data/charset_size.txt \
        --data_dir=/mnt/data/datasets/ \
        --output_tfrecord=/mnt/data/datasets/tfexample_train  \
        --text_length=20
```

Upon some data exploration, it was found that the maximum length of text in our dataset was 31. So, the model was trained with `text_length` set to 31. But the resulting model was very inaccurate. With some more experiments, we found that the number of examples with the length of text more than 20 were 10. In other words, for most of the samples there were more null characters than the alphabets. 

### 2 - Train

If you want to fine tune:

```bash
wget http://download.tensorflow.org/models/attention_ocr_2017_08_09.tar.gz && \
tar xf attention_ocr_2017_08_09.tar.gz 
```

Export path for our custom.py dataset script:

```bash
export PYTHONPATH=$PYTHONPATH:./datasets/
```

```python3
python train.py \
    --dataset_name=custom \
    --batch_size=1 \
    --train_log_dir=/mnt/output/ \
    --checkpoint=model.ckpt-399731
    --max_number_of_steps=110000
```

### 3 - Run inference

```python3
python3 demo_inference.py --dataset_name=custom --batch_size=1 --image_path_pattern=/data/trd/ --checkpoint=/data/model.ckpt-110000
```

### 4 - Evaluation

```python3
python3 eval.py --dataset_name=custom --batch_size=1 --split_name=test --num_batches=9 --train_log_dir=/data/
```
