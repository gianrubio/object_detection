from string import Template
import os

xml_label_template = Template(
    """
    <annotation>
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
    with open(os.path.join(path, file_name), "w") as f:
        f.write(
            xml_label_template.substitute(
                label=label,
                file_name=file_name,
                path=path,
                width=width,
                height=height,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
