# emotion detection
## trained models
| model name  | test accuracy | frame time | dataset |
| ------------- | ------------- | ------------- | ------------- |
| emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5 | 0.66  | 7ms | FER |
| emotion_mini_XCEPTION_48x48_0.63_5ms.hdf5 | 0.63  | 5ms | FER |
* frame time here is running on Keras on python and a Macbook CPU. Tensorflow + CPU is ~2ms faster. 

## model conversions
- [x] from keras to tensorflow frozen graph (needs https://github.com/azhongwl/keras_to_tensorflow)

- [x] from frozen graph to tflite float model
```sh
bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- --input_file=<input: frozen graph> --output_file=<output: .tflite model> --inference_type=FLOAT --input_shape=1,64,64,1 --input_array=input_1 --output_array=output_node0
```

- [ ] from frozen graph to tflite quantized model

## TODO
- [ ] train with FER+ data labelled with 10 people, (i.e. each image will have 10 votes)
- [ ] train with SVM on the last stage of network
