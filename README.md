# FACE SWAPPER
Have you ever wanted to know how your face would look in someone else's head? This program takes an image with two visible faces on it, and swaps them to create some funny results.

## Dependencies:
* [Python](https://www.python.org/doc/) - 3.10.5
* [OpenCV](https://docs.opencv.org/4.6.0/) - 4.6.0
* [Numpy](https://numpy.org/doc/stable/) - 1.22.4

## How to use:
1. Copy the image you want to work with inside the */visuals* folder.
2. Go to the */src* folder and execute the main program:
```console
    $ python FaceSwapper.py
```
3. If everything went fine, a new window will pop up showing the resulting image. Just press any key to close the window and save the new image in the same */visuals* folder. The name will be `swapped-aaa.xxx`, where `aaa` corresponds to the original image name and `xxx` its extension.

For example:

![alt text](https://github.com/Josgonmar/Face-swapper/blob/master/docs/test.jpeg?raw=true)

It's converted into:

![alt text](https://github.com/Josgonmar/Face-swapper/blob/master/docs/swapped-test.jpeg?raw=true)

As you can see, both faces are swapped within the same image. First it uses a deep learning model to detect both bounding boxes with high precision, to later use [OpenCV's](https://docs.opencv.org/4.6.0/d2/d42/tutorial_face_landmark_detection_in_an_image.html) face landmark detector model. With these landmarks, several affine warping operations are performed using delauny triangles, as well as some morphological operations, in order to get the final results.

Of course, the results may differ depending on the source image and the clarity of the faces on it.
## License:
Feel free to use this programa whatever you like!