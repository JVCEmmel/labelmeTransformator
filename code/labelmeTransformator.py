import os
import sys
import json
import shutil
import numpy as np

class Image:

    __doc__ = "The class, representing the image-data for the COCO Dataset"

    def __init__(self, id, name, height, width):
        self.id = id
        self.name = name
        self.height = height
        self.width = width
    
    def convertToDictionary(self):
        imagedict = {}
        imagedict["height"] = self.height
        imagedict["width"] = self.width
        imagedict["id"] = self.id
        imagedict["name"] = self.name

        return imagedict


class Category:

    __doc__ = "The class, representing the categorie-data for the COCO Dataset"

    def __init__(self, id, supercategory, name):
        self.id = id
        self.name = name
        self.supercategory = supercategory
    
    def convertToDictionary(self):
        categorydict = {}
        categorydict["supercategory"] = self.supercategory
        categorydict["id"] = self.id
        categorydict["name"] = self.name

        return categorydict


class Polygon:
    
    __doc__ = "The class, representing the categorie-data for the COCO Dataset"

    def __init__(self, id, category_id, image_id, iscrowd, segmentation, bbox, area):
        self.id = id
        self.category_id = category_id
        self.image_id = image_id
        self.iscrowd = iscrowd
        self.segmentation = segmentation
        self.bbox = bbox
        self.area = area

    def convertToDictionary(self):
        polygondict = {}
        polygondict["segmentation"] = self.segmentation
        polygondict["iscrowd"] = self.iscrowd
        polygondict["area"] = self.area
        polygondict["image_id"] = self.image_id
        polygondict["bbox"] = self.bbox
        polygondict["category_id"] = self.category_id
        polygondict["id"] = self.id

        return polygondict


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


# PROGRAM START
# ------------- 
if __name__ == "__main__":
    # helping variables
    # -----------------
    # path gives the directory with images and .json,
    # images, categories and polygons store the classes created,
    # label_list and polygon_list keep track over the gathered labels and polygonids
    path = sys.argv[1]
    json_dump = {}
    images = []
    categories = []
    polygons = []
    label_list = {}
    polygon_list = []

    # get all json files in directory
    json_list = sorted([f for f in os.listdir(path) if f.endswith(".json")])

    # looping through the .json files
    # 
    # the first loop saves the general image data
    # the first enclosed loop saves the label data
    # the second enclosed loop saves the coordinates of the poligons
    for id_count, json_file in enumerate(json_list):
        with open(path + '/' + json_file, "r") as content:
            data = json.load(content)
            
            image = Image(id_count, json_file[:-4] + "jpg", data["imageHeight"], data["imageWidth"])
            images.append(image.convertToDictionary())

            for shape_count, element in enumerate(data["shapes"]):
                if element["label"] not in label_list:
                    category = Category((len(label_list)), None, element["label"])
                    categories.append(category.convertToDictionary())
                    label_list[element["label"]] = (len(label_list))

                x_coordinates = []
                y_coordinates = []

                # extract the polgon points
                for polygon in element["points"]:
                    x_coordinates.append(polygon[0])
                    y_coordinates.append(polygon[1])

                # transform into COCO format
                segmentation = list(sum(zip(x_coordinates, y_coordinates), ()))

                # get the values of the bbox
                smallest_x = int(min(x_coordinates))
                smallest_y = int(min(y_coordinates))
                biggest_x = int(max(x_coordinates))
                biggest_y = int(max(y_coordinates))

                bbox_height = biggest_y-smallest_y
                bbox_width = biggest_x-smallest_x

                bbox = [smallest_x, smallest_y, bbox_width, bbox_height]

                # get the area of the polygon
                polygon_area = PolyArea(x_coordinates, y_coordinates)

                # create polygon instance and add it to the list
                polygon = Polygon(len(polygon_list), label_list[element["label"]], id_count, 0, segmentation, bbox, polygon_area)
                polygons.append(polygon.convertToDictionary())
                polygon_list.append(shape_count)

    # fill the dictionary to dump the data
    json_dump["images"] = images
    json_dump["categories"] = categories
    json_dump["annotations"] = polygons

    # restructure the files (This part is optional and can be deleted)
    # ---------------------
    # create destination directory
    if os.path.isdir(path + "/labelme_jsons") == False:
        os.makedirs(path + "/labelme_jsons")

    # copy pictures
    [shutil.move(path + "/" + element, path + "/labelme_jsons/" + element) for element in json_list]

    # final dump
    with open(path + "/output.json", "w") as output_file:
        json.dump(json_dump, output_file, indent=4)
