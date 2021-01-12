import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        # print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    parser = argparse.ArgumentParser(
        description="Partition dataset of images into training and testing sets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--imageDir",
        help="Path to the folder where the image dataset is stored. If not specified, the CWD will be used.",
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        help="Path to the output folder where the train and test dirs should be created. "
        "Defaults to the same directory as IMAGEDIR.",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    xml_df = xml_to_csv(os.path.join(args.imageDir))
    xml_df.to_csv(args.outputDir, index=None)
    print(f"Successfully converted xml to csv {args.outputDir}.")


main()
