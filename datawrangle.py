# Python Imports
import os 
import cv2
import imutils
import time
import json

from cityscapesscripts.helpers.annotation import CsBbox3d
from cityscapesscripts.helpers.box3dImageTransform import (
    Camera, 
    Box3dImageTransform,
    CRS_V,
    CRS_C
)


class CityScapeDataset:
    def __init__(self, ):
        # PATHS
        ## All the images from the dataset.
        ## Folder structure "images-->{type}-->{city}-->{[list of images]}"
        self.IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images")

        ## All the different .JSON label from the dataset.
        ## Folder structure "labels-->{"people", "vehcile"}-->{type}-->{city}-->{[list of json]}"
        self.LABEL_PATH = os.path.join(os.path.dirname(__file__), "labels")

        ## METADATA about the test vehicle in the .JSON from dataset.
        ## Folder structure "testvec_metadata-->{type}-->{city}-->{[list of json]}"
        self.VECDATA_PATH = os.path.join(os.path.dirname(__file__), "testvec_metadata")

        ## CAMERA properties for each images in the .JSON from dataset
        ## Folder structure "camera-->{type}-->{city}-->{[list of json]}" 
        self.CAMERA_PATH = os.path.join(os.path.dirname(__file__), "camera")
        
        # Sanity checks
        self.__check_path(self.IMAGE_PATH)
        self.__check_path(self.LABEL_PATH)
        self.__check_path(self.VECDATA_PATH)
        self.__check_path(self.CAMERA_PATH)

    def __check_path(self, path):
        if not os.path.exists(path):
            raise Exception("The PATH does not exit!\nPATH: {}".format(path))


    def get_imagepath_list(self, type, city):
        """
        Function to get the list of images for the specific type (train, val, test) and the city"

        INPUT:
            type(str):  Type of images to load -> train, val or test.
            city(str):  Imgae for the city to load.
        RETURN
            <list>
            Returns the list of full path of images for the specific type and city.
        """
        # Folder path
        folder_path = os.path.join(self.IMAGE_PATH, type, city)
        self.__check_path(folder_path)

        return [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    def get_label(self, image_path, label_type):
        """
        Function to get the label for sepecific "images_path" in the dataset.

        INPUT:
            image_path(str):  Full path of the image for which the .JSON is loaded.
            label_type(str):  Type of label to load. Either "people" or "vehicle".
        RETURN
            <dict>
            Returns the .JSON loaded for the label_type, type and city.
        """
        # Type and City
        temp = image_path.split("/")
        type = temp[len(temp) - 3]
        city = temp[len(temp) - 2]
        folder_path = os.path.join(self.LABEL_PATH, label_type, type, city)

        file_name = str()
        if label_type=="people":
            file_name = "_".join(os.path.basename(image_path).split(".")[0].split("_")[:3]) + "_gtBboxCityPersons.json"
        elif label_type=="vehicle":
            file_name = "_".join(os.path.basename(image_path).split(".")[0].split("_")[:3]) + "_gtBbox3d.json"

        # Complete path
        label_path = os.path.join(folder_path, file_name)
        self.__check_path(label_path)

        with open(label_path) as json_file:
            return json.load(json_file)

    def get_testvechile_data(self, image_path):
        # Type and City
        temp = image_path.split("/")
        type = temp[len(temp) - 3]
        city = temp[len(temp) - 2]
        
        folder_path = os.path.join(self.VECDATA_PATH, type, city)
        file_name = "_".join(os.path.basename(image_path).split(".")[0].split("_")[:3]) + "_vehicle.json"
        # Complete path
        label_path = os.path.join(folder_path, file_name)
        self.__check_path(label_path)

        with open(label_path) as json_file:
            return json.load(json_file)

    def get_camera_paramters(self, image_path):
        # Type and City
        temp = image_path.split("/")
        type = temp[len(temp) - 3]
        city = temp[len(temp) - 2]
        
        folder_path = os.path.join(self.CAMERA_PATH, type, city)
        file_name = "_".join(os.path.basename(image_path).split(".")[0].split("_")[:3]) + "_camera.json"
        # Complete path
        label_path = os.path.join(folder_path, file_name)
        self.__check_path(label_path)

        with open(label_path) as json_file:
            return json.load(json_file)

    def get_object_coordinates(self, sensor_meta, object_meta, coordinate_type="vehicle"):
        # Create a instance of camera using the intresic+extransic parameters 
        # from the annotation
        camera = Camera(fx=sensor_meta["fx"],
                        fy=sensor_meta["fy"],
                        u0=sensor_meta["u0"],
                        v0=sensor_meta["v0"],
                        sensor_T_ISO_8855=sensor_meta["sensor_T_ISO_8855"])
        # Using the camera, then create a instance for coordinate transormation
        box3d_annotation = Box3dImageTransform(camera=camera)

        # Class to load the annotation data from the "BBOX"
        # This is a function from where we fetch bbox (amodel) and label
        obj = CsBbox3d()
        # Create class object for the first "object" in the meta data
        obj.fromJsonText(object_meta)
        # Init the box3D
        box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V)

        if coordinate_type.lower() == "vehicle":
            return box3d_annotation.get_vertices(coordinate_system=CRS_V)
        elif coordinate_type.lower() == "car":
            return box3d_annotation.get_vertices(coordinate_system=CRS_C)
        elif coordinate_type.lower() == "image":
            return box3d_annotation.get_vertices_2d()
        else:
            return None
            

    
if __name__ == "__main__":
    cs = CityScapeDataset()
    
    city = "aachen"
    dataset_type = "train"

    # Fetch the image path from the list
    image_path_list = cs.get_imagepath_list(dataset_type, city)
    print("\nNumber of image for {} in {} dataset: {}".format(city, dataset_type, len(image_path_list)))

    # Choose an image
    image_path = image_path_list[5]
    
    # Fetch the labeled META for the specific image. 
    image_meta = cs.get_label(image_path, label_type="vehicle")
    print("\nKey in the JSON image meta data: ", image_meta.keys())
    print("\nObjects in the image:")
    print([objects["label"] for objects in image_meta["objects"]])

    # Fetch test vehicle data
    vec_meta = cs.get_testvechile_data(image_path)
    print("\nKey in the JSON test vehicle data: ", vec_meta.keys())

    # Fetch camera parameters for the image
    camera_paramters = cs.get_camera_paramters(image_path)
    print("\nKey in the JSON test camera data: ", camera_paramters.keys())
    print("CAMERA META DATA")
    for k, v in camera_paramters.items():
        print("{}: {}".format(k, v))

    # Get 3D coordinate of the object in vehcile and image plane
    # This is only applicable for label_type="vehicle"

    # Print the vertices of the box.
    # loc is encoded with a 3-char code
    #   0: B/F: Back or Front
    #   1: L/R: Left or Right
    #   2: B/T: Bottom or Top
    # BLT -> Back left top of the object
    object = image_meta["objects"][0]
    print("\n Cooddinate of {} in Vehicle plane:".format(object["label"]))
    coordinates = cs.get_object_coordinates(image_meta["sensor"], object,  coordinate_type="vehicle")
    print("     {:>8} {:>8} {:>8}".format("x[m]", "y[m]", "z[m]"))
    for loc, coord in coordinates.items():
        print("{}: {:8.2f} {:8.2f} {:8.2f}".format(loc, coord[0], coord[1], coord[2]))

    # Print the vertices of the box.
    # loc is encoded with a 3-char code
    #   0: B/F: Back or Front
    #   1: L/R: Left or Right
    #   2: B/T: Bottom or Top
    # BLT -> Back left top of the object
    print("\n Cooddinate of {} in Image plane:".format(object["label"]))
    coordinates = cs.get_object_coordinates(image_meta["sensor"], object,  coordinate_type="image")
    print("\n     {:>8} {:>8}".format("u[px]", "v[px]"))
    for loc, coord in coordinates.items():
        print("{}: {:8.2f} {:8.2f}".format(loc, coord[0], coord[1]))

    # Draw rectangle on the image for the object
    image = cv2.imread(image_path)
    xmin = int(min([p[0] for p in coordinates.values()]))
    ymin = int(min([p[1] for p in coordinates.values()]))
    xmax = int(max([p[0] for p in coordinates.values()]))
    ymax = int(max([p[1] for p in coordinates.values()]))
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    
    # Show image
    cv2.imshow("Chossen Image", imutils.resize(image, width=1280))
    time.sleep(0.1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    