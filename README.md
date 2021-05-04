# Cityscape-dataset-wrangler
Data wrangler for Cityscape Dataset. 

# Folder Structure
You need to have the data from the dataset into following folder structure. 
- All the **images** in a folder "images" with folder tree: "images"-->{type}-->{city}-->{[list of images]}.
- All the **image meta** in a folder "labels" with folder tree: "labels"-->{"people", "vehcile"}-->{type}-->{city}-->{[list of json]}.
- All the **vehicle meta** in a folder "testvec_metadata" with folder tree: "testvec_metadata"-->{type}-->{city}-->{[list of json]}.

# Functions
- get_imagepath_list(type, city):           Get the list of full path of images for specific city and data type
- get_label(image_path, label_type):        Get label meta for each specific image.
- get_testvechile_data(image_path):         Get test vehicle meta data for the image. 

# Usage
Check the file datawrange.py on how to use the functions. 

### Reference
- [Cityscape dataset](https://www.cityscapes-dataset.com)



