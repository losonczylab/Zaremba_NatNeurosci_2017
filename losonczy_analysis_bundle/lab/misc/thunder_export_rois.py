"""Export SIMA ROIs to Thunder ROIs JSON"""

import json


def convert(dataset, rois_label, path, plane=0):

    rois = dataset.ROIs[rois_label]

    out = []

    for roi in rois:
        coords = []
        values = []
        for coord, val in roi.mask[plane].todok().iteritems():
            coords.append([int(coord[0]), int(coord[1])])
            values.append(int(val))
        out.append({'id': roi.id, 'coordinates': coords, 'values': values})

    json.dump(
        out, open(path, 'w'), sort_keys=False, indent=None, separators=(', ', ': '))
