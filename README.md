# DispNet-TensorFlow
This project is no longer under active development, so please exercise caution when using it and I hope it can still be helpful to you :) 

TensorFlow implementation of [A Large Dataset to Train Convolutional Networks
for Disparity, Optical Flow, and Scene Flow Estimation](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mayer_A_Large_Dataset_CVPR_2016_paper.pdf) by Zhijian Jiang.

## Dataset
* [Scene Flow Datasets: FlyingThings3D, Driving, Monkaa](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

## Tutorials
### TensorFlow
* [Tensorflow tutorials (Eng Sub) 神经网络 教学 教程](https://www.youtube.com/watch?v=RSRkp8VAavQ&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)

## Network
### Convolutional Network

|Name | Kernel | Strides | Channels I/O | Input Resolution | Output Resolution | Input |
|--- | --- | --- | --- | --- | --- | --- |
|conv1 	  	| 7 * 7 | 1 | 6/64 		| 1536 * 768 	| 768 * 384 | Images |
|max_pool1 	| 2 * 2 | 2 | 64/64 	| 1536 * 768 	| 768 * 384 | conv1 | 
|conv2 	  	| 5 * 5 | 1 | 64/128 	| 768 * 384		| 384 * 192 | max_pool1 |
|max_pool2 	| 2 * 2 | 2 | 128/128 	| 768 * 384		| 384 * 192 | conv2|
|conv3a 	  	| 5 * 5 | 1 | 128/256 	| 384 * 192		| 192 * 96 	| max_pool2|
|max_pool3 	| 2 * 2 | 2 | 256/256 	| 384 * 192		| 192 * 96 	| conv3a|
|conv3b 		| 3 * 3 | 1 | 256/256	| 192 * 96		| 192 * 96	| max_pool3|
|conv4a 	  	| 5 * 5 | 1 | 256/512 	| 192 * 96		| 96 * 48 	| conv3b|
|max_pool4 	| 2 * 2 | 2 | 512/512 	| 96 * 48		| 96 * 48 	| conv4a|
|conv4b 		| 3 * 3 | 1 | 512/512	| 96 * 48		| 96 * 48	| max_pool4|
|conv5a 	  	| 5 * 5 | 1 | 512/512 	| 96 * 48		| 48 * 24 	| conv4b|
|max_pool5 	| 2 * 2 | 2 | 512/512 	| 48 * 24		| 48 * 24 	| conv5a|
|conv5b 		| 3 * 3 | 1 | 512/512	| 48 * 24		| 48 * 24	| max_pool5|
|conv6a 	  	| 5 * 5 | 1 | 512/512 	| 48 * 24		| 24 * 12	| conv5b|
|max_pool6 	| 2 * 2 | 2 | 1024/1024 | 24 * 12		| 24 * 12 	| conv6a|
|conv6b 		| 3 * 3 | 1 | 1024/1024	| 24 * 12		| 24 * 12	| max_pool6|
|pr6 + loss6	| 3 * 3	| 1	| 1024/1	| 24 * 12		| 24 * 12	| conv6b|

### Upconvolutional Network

|Name | Kernel | Strides | Channels I/O | Input Resolution | Output Resolution | Input |
|--- | --- | --- | --- | --- | --- | ---|
|upconv5		| 4 * 4	| 2 | 1024/512	| 24 * 12		| 48 * 24	| conv6b|
|iconv5		| 3 * 3	| 1	| 1024/512	| 48 * 24		| 48 * 24	| upconv5 + conv5b|
|pr5+loss5	| 3 * 3 | 1	| 512/1		| 48 * 24		| 48 * 24	| iconv5|
|upconv4		| 4 * 4	| 2 | 512/256	| 48 * 24		| 96 * 48	| iconv5|
|iconv4		| 3 * 3	| 1	| 768/256	| 96 * 48		| 96 * 48	| upconv4 + conv4b|
|pr4+loss4	| 3 * 3 | 1	| 512/1		| 96 * 48		| 96 * 48	| iconv4|
|upconv3		| 4 * 4	| 2 | 256/128	| 96 * 48		| 192 * 96	| iconv4|
|iconv3		| 3 * 3	| 1	| 384/128	| 192 * 96		| 192 * 96	| upconv3 + conv3b|
|pr3+loss3	| 3 * 3 | 1	| 128/1		| 192 * 96		| 192 * 96	| iconv3|
|upconv2		| 4 * 4	| 2 | 128/64	| 192 * 96		| 384 * 192	| iconv3|
|iconv2		| 3 * 3	| 1	| 192/64	| 384 * 192		| 384 * 192	| upconv2 + conv2|
|pr2+loss2	| 3 * 3 | 1	| 64/1		| 384 * 192		| 384 * 192	| iconv2|
|upconv1		| 4 * 4	| 2 | 64/32		| 384 * 192		| 768 * 384	| iconv2|
|iconv1		| 3 * 3	| 1	| 96/32		| 768 * 384		| 768 * 384	| upconv1 + conv1|
|pr1+loss1	| 3 * 3 | 1	| 32/1		| 768 * 384		| 768 * 384	| iconv1|

## Issues
* How to input png images:   
	* [Solution 1 -- FAIL](https://github.com/tensorflow/models/issues/564):  
	```
	contents = ''
	with open('path/to/image.jpeg') as f:
   		contents = f.read()
	tf.image.decode_jpeg(contents) 
	```

	* [Solution 2 -- FAIL](http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels)
	```
	reader = tf.WholeFileReader(http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels)
  	key, value = reader.read(filename_queue)
  	example = tf.image.decode_png(value)
  	```

  	* [Solution 3 -- Success](http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels)
  	```
  	file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
