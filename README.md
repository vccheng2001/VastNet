# VASTNet
**ECE Capstone Project Fall 2021**

The esp32cam is an Arduino library that provides an easy-to-use API with the OV2640 camera on the ESP32 microcontroller. 
We use a pre-trained Yolov3-tiny model for real-time object detection: a compressed version of Yolov3 to improve the model's efficiency and reduce its memory footprint.

## Installation 
------- 
Clone repository:

```git clone vastnet ```

Run real-time inference.
To view argument options add the ```--help``` flag.

```python3 inference.py --model_name=<model_name> ... ```



## Files
-----

- Model config files are stored as ```darknet/cfg/<model_name>.cfg```
- Pre-trained weights files are stored as ```darknet/cfg/<model_name>.weights```
- Class names are stored in ```darknet/data/coco.names```


Model
------
| Model | Inference time | 
| ------ | ----------- | 
| yolov3   | ~400-600ms |
| yolov3-tiny   | ~40-60ms | 



## Useful Links 
------
- [Darknet Github](https://github.com/pjreddie/darknet)

- [Yolo-v3](https://pjreddie.com/darknet/yolo/)

