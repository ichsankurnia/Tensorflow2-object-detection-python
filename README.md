# Getting Started

## Setup Virtual Environment

`python -m venv env`

`source env/bin/activate` # Linux

`.\env\Scripts\activate` # Windows

`deactivate` # exit from virtual environment


### Optional if use jupyter

`pip install ipykernel jupyter`

`python -m ipykernel install --user --name=env`



## LABELIMG

`pip install opencv-python`

collect image, store to images folder, 20% for folder test and 80% for folder train in images folder

`pip install --upgrade pyqt5 lxml`

`git clone https://github.com/tzutalin/labelImg`

`cd labelImg && pyrcc5 -o libs/resources.py resources.qrc`

`cd labelImg && python labelImg.py (label the images from folder train n test)`



## Setup Project

`mkdir Project`


copy images folder after labelImg to Project

`cd Project`

`git clone https://github.com/tensorflow/models.git`

download n extract protoc [https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip](https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip)

copy protoc.exe to models/research



## INSTALLATIONS

`cd models\research`


### Compile protos.

`protoc object_detection/protos/*.proto --python_out=.`


### Install TensorFlow Object Detection API.

`cp object_detection/packages/tf2/setup.py .`

`python -m pip install .`


### Test the installation.

`python object_detection/builders/model_builder_tf2_test.py`
#Ran 24 tests in 28.539s  OK (skipped=1), if not like this, install the missing module base on console

python and type import object_detection then enter

*if error, install the missing module base on console



## GENERATE TFRECORD

cd to root project, `mkdir data`, create label_map.pbtxt in data folder

adjust label_map.pbtxt with label name such as => item { name:'car' id:1 } item { name:'handphone' id:2 }

cd to root project, `git clone https://github.com/nicknochnack/GenerateTFRecord`

`python GenerateTFRecord\generate_tfrecord.py -x images\train -l data\label_map.pbtxt -o data\train.record`

`python GenerateTFRecord\generate_tfrecord.py -x images\test -l data\label_map.pbtxt -o data\test.record`



## TRAIN MODEL

download and extract zoo model in [SSD MobileNet V2 FPNLite 320x320](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

`mkdir mymodels`

copy pipeline.config in zoo model to mymodels


### Update Config For Transfer Learning

adjust pipline.config in mymodels (num_class, batch_size, fine_tune_checkpoint, fine_tune_checkpoint_type="detection" label_map_path, input_path) 

or :

adjust config in script\update-pipline-config.py then run `python script\update-pipline-config.py`


### Start Training

`python models\research\object_detection\model_main_tf2.py --model_dir=mymodels --pipeline_config_path=mymodels\pipeline.config --num_train_steps=2000`



## --> EVALUATE MODEL

`python models\research\object_detection\model_main_tf2.py --model_dir=mymodels --pipeline_config_path=mymodels\pipeline.config --checkpoint_dir=mymodels`

`cd mymodels\train && tensorboard --logdir=.`



### ISSUE in opencv-python

`pip uninstall opencv-python-headless -y`

`pip uninstall opencv-python -y`


`pip install opencv-python`



## --> TEST DETECTION with Load Train Model From Checkpoint

`python script\ckpt-detect-img-cv2.py`

`python script\ckpt-detect-video.py`



## Freezing the Graph

`cd mymodels && mkdir export`

then, in root projcet run:

`python models\research\object_detection\exporter_main_v2.py  --input_type=image_tensor --pipeline_config_path=mymodels\pipeline.config --trained_checkpoint_dir=mymodels --output_directory=mymodels\export`



## EXPORT INFERENCE_GRAPH FOR Tensorflow 1.X

`python models\research\object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path mymodels\pipeline.config --trained_checkpoint_prefix mymodels\model.ckpt-3 --output_directory mymodels\inference_graph`

*note adjust model.ckpt-{highestCount}

*node this script not work in TF2



## Export TO TFJS USE FOR WEB APP

`pip install tensorflowjs`

`cd mymodels && mkdir tfjsexport`

then, in root projcet run:

`tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default mymodels\export\saved_model mymodels\tfjsexport`



## CONVERTION TO TFLITE USE FOR MOBILE OR RASPY

`cd mymodels && mkdir tfliteexport`

then, in root projcet run:

`python models\research\object_detection\export_tflite_graph_tf2.py  --pipeline_config_path=mymodels\pipeline.config --trained_checkpoint_dir=mymodels --output_directory=mymodels\tfliteexport`

`tflite_convert --saved_model_dir=mymodels\tfliteexport\saved_model --output_file=mymodels\tfliteexport\saved_model\detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops`



## --> TEST DETECTION FROM EXPORTED MODEL <--

`python script\image-detection.py`                      # for image detection

`python script\realtime-detection.py`                   # for realtime (webcam or video) detection