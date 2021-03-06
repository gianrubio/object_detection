# xml to csv
import cv2
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import json
import base64

file_path = "/Users/grubio/Downloads/image/multiple-image-detection/models/research/object_detection/trainning/images/train"


def xml2csv(xml_path):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

    """
    print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df = pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height',
                           'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    return xml_df


def df2labelme(symbolDict, image_path, image):
    """ convert annotation in CSV format to labelme JSON

    Args:
        symbolDict (dataframe): annotations in dataframe
        image_path (str): path to image
        image (np.ndarray): image read as numpy array

    Returns:
        JSON: converted labelme JSON

    """
    try:
        symbolDict['min'] = symbolDict[['xmin', 'ymin']].values.tolist()
        symbolDict['max'] = symbolDict[['xmax', 'ymax']].values.tolist()
        symbolDict['points'] = symbolDict[['min', 'max']].values.tolist()
        symbolDict['shape_type'] = 'rectangle'
        symbolDict['group_id'] = None
        height, width, _ = image.shape
        symbolDict['height'] = height
        symbolDict['width'] = width
        encoded = base64.b64encode(open(image_path, "rb").read())
        symbolDict.loc[:, 'imageData'] = encoded
        symbolDict.rename(columns={'class': 'label', 'filename': 'imagePath',
                                   'height': 'imageHeight', 'width': 'imageWidth'}, inplace=True)
        converted_json = (symbolDict.groupby(['imagePath', 'imageWidth', 'imageHeight', 'imageData'], as_index=False)
                          .apply(lambda x: x[['label', 'points', 'shape_type', 'group_id']].to_dict('r'))
                          .reset_index()
                          .rename(columns={0: 'shapes'})
                          .to_json(orient='records'))
        return json.loads(converted_json)[0]
    except Exception as e:
        print(f' conversion to json2me failed for {image_path}. Error {e}')
        raise e


for path, dirs, files in os.walk(file_path):
    for file in files:
        file_name, extension = os.path.splitext(file)
        if extension != ".xml":
            continue
        full_file_path = os.path.join(file_path, file)
        xml_csv = xml2csv(full_file_path)

        image_path = os.path.join(file_path, xml_csv.iloc[0].filename)
        image = cv2.imread(image_path)
        csv_json = df2labelme(xml_csv, image_path, image)
        json_file_path = os.path.join(file_path, f"{file_name}.json")
        with open(json_file_path, 'w') as f:
            print(f"writing {json_file_path}")
            f.write(json.dumps(csv_json))
