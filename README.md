# labelmeTransformator

This short program was designed to transform json-files, created with the "labelme" software by [wkentaro](https://github.com/wkentaro/labelme), into a COCO Dataset format.
The program is inspired by [the tutorial](https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/) of Tony607 and the final form of my COCO structure is orientated at his.

## Usage

This program needs one argument. To run via console type "python labelmeTransformator.py /path/to/image/folder"

In the directory should be the images and the json files created with labelme.
