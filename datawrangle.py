# Python Imports
import os 
import cv2
import imutils
import time
import json

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

        # Sanity checks
        self.__check_path(self.IMAGE_PATH)
        self.__check_path(self.LABEL_PATH)
        self.__check_path(self.VECDATA_PATH)

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
    image_meta = cs.get_label(image_path, label_type="people")
    print("\nKey in the JSON image meta data: ", image_meta.keys())
    print("Objects in the image:")
    print([objects["label"] for objects in image_meta["objects"]])

    # Fetch test vehicle data
    vec_meta = cs.get_testvechile_data(image_path)
    print("\nKey in the JSON test vehicle data: ", vec_meta.keys())

    # Show image
    image = cv2.imread(image_path)
    cv2.imshow("Chossen Image", imutils.resize(image, width=1280))
    time.sleep(0.1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()