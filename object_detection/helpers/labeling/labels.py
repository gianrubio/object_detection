from string import Template
import os

xml_label_template = Template(
    """<annotation>
        <folder>test</folder>
        <filename>$file_name</filename>
        <path>$path</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>$width</width>
            <height>$height</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>$label</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>$xmin</xmin>
                <ymin>$ymin</ymin>
                <xmax>$xmax</xmax>
                <ymax>$ymax</ymax>
            </bndbox>
        </object>
    </annotation>
"""
)


def generate_file_labels(label, file_name, path, width, height, xmin, ymin, xmax, ymax):
    if xmin == 0:
        xmin = 1
    if ymin == 0:
        ymin = 1

    with open(os.path.join(path, f"{file_name}.xml"), "w") as f:
        f.write(
            xml_label_template.substitute(
                label=label,
                file_name=f"{file_name}.jpg",
                path=path,
                width=width,
                height=height,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
