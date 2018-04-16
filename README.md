# emotion detection
## trained models
| model name  | test accuracy | frame time | dataset |
| ------------- | ------------- | ------------- | ------------- |
| emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5 | 0.66  | 7ms | FER |
| emotion_mini_XCEPTION_48x48_0.63_5ms.hdf5 | 0.63  | 5ms | FER |

## TODO
1. train with FER+ data labelled with 10 people, (i.e. each image will have 10 votes)
2. train with SVM on the last stage of network
