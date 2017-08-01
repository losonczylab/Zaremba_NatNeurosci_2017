"""Note -- user needs to install nd2reader module:
   pip install nd2reader --user
"""

import os
from os.path import splitext, join
import argparse
import numpy as np
import nd2reader
from sima.misc.tifffile import imsave

from pudb import set_trace


def nd2_to_tiff(data_path):
    basename = splitext(data_path)[0]
    nd2 = nd2reader.Nd2(data_path)

    frames = nd2.frames
    channels = nd2.channels
    planes = nd2.z_levels
    fovs = nd2.fields_of_view

    for f in frames:
        for c in channels:
            for p in planes:
                for fov in fovs:
                    output_filename = basename + '_frame_' + str(f) + \
                        '_channel_' + c + '_plane_' + str(p) + '_field_' + \
                        str(fov) + '.tif'
                    try:
                        image = nd2.get_image(
                            frame_number=f, field_of_view=fov,
                            channel_name=c, z_level=p)
                    except:
                        import pudb; pudb.set_trace()
                    image = np.array(image).astype('uint16')
                    imsave(filename=output_filename, data=image)


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d", "--directory", action="store", type=str, default='',
        help="Process the t-series folders contained in 'directory'")
    args = argParser.parse_args()

    for f in os.listdir(args.directory):
        if f.endswith('.nd2'):
            data_path = join(args.directory, f)
            print 'converting {p}...'.format(p=data_path)
            nd2_to_tiff(data_path)
