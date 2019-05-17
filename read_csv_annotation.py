import pandas as pd
import glob
import xml.etree.ElementTree as ET

xml_folder = 'D:/Vinoj/HandsOnCV/object_detection/FRCNN/BCCD_Dataset-master/BCCD/Annotations/'
xml_path_list = glob.glob(xml_folder + '*.xml')


def xml2dataframe(xmlpath):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    df = pd.DataFrame()
    image_name = root.find('filename').text
    for object_itr in root.findall('object'):
        class_name = object_itr.find('name').text
        bndbox = object_itr.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        series = {'image_name': image_name, 'class_name': class_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax,
                  'ymax': ymax}
        df = df.append(series, ignore_index=True)
    return df


master_df = pd.DataFrame()
for xml in xml_path_list:
    df = xml2dataframe(xml)
    master_df = master_df.append(df)

print("Number of Images:", master_df.image_name.nunique())
print("Number of Bounding Boxes:", len(master_df))


master_df.to_csv('full_dataset.csv')

import sklearn.tests
