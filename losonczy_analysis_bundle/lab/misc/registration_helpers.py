import os
from os.path import join, dirname
from glob import glob

from xml.etree import ElementTree
import numpy as np

from lab.misc.auto_helpers import locate, get_element_size_um, get_prairieview_version


def get_plane_spacing(xml_path):

    """Determine the spacing in microns between planes in a T/Z-series."""

    parser = ElementTree.iterparse(xml_path, ['start', 'end'])
    parser = iter(parser)
    _, root = parser.next()
    seq = root.find('Sequence')

    def get_frame_pos(frame):

        try:
            for state in frame.findall('PVStateShard'):
                for state_val in state.findall('PVStateValue'):
                    if state_val.get('key') == 'positionCurrent':
                        for axis in state_val.findall('SubindexedValues'):
                            if axis.get('index') == 'ZAxis':
                                for pos in axis.findall('SubindexedValue'):
                                    if pos.get('description').endswith('Piezo'):
                                        return float(pos.get('value'))
        except TypeError:
            return None

    idx = None
    pos = None

    # We need to find a pair of adjacent frames for which the piezo position
    # was recorded. Sometimes frames are skipped or piezo info is skipped.
    MAX_SEARCH = 20  # Only look through at most first 20 frames

    for frame in seq.findall('Frame')[:MAX_SEARCH]:

        if not idx:
            pos = get_frame_pos(frame)
            # If a piezo position is present, save the frame idx
            if pos:
                idx = int(frame.get('index'))
        else:
            newidx = int(frame.get('index'))
            newpos = get_frame_pos(frame)

            if newpos:
                if newidx - idx == 1:
                    # If this is the next frame, we are done
                    return np.abs(newpos - pos)
                    # If not, start over with this frame
                else:
                    idx = newidx
                    pos = newpos

    return 0


def get_z_buffer(t_xml, z_xml):

    raise('Not Implemented')

    t_tree = ElementTree.parse(t_xml)
    seq = t_tree.getroot().find('Sequence')

    # Due to unreliability in recording 1st frame info,
    # grab 2nd and 3rd and then confirm index
    frame_1 = seq.findall('Frame')[1]
    frame_2 = seq.findall('Frame')[2]

    plane = int(frame_1.get('index'))

    def get_frame_pos(frame):

        for state in frame.findall('PVStateShard'):
            for state_val in state.findall('PVStateValue'):
                if state_val.get('key') == 'positionCurrent':
                    for axis in state_val.findall('SubindexedValues'):
                        if axis.get('index') == 'ZAxis':
                            for pos in axis.findall('SubindexedValue'):
                                if pos.get('description').endswith('Piezo'):
                                    return float(pos.get('value'))

    piezo_pos_1 = get_frame_pos(frame_1)
    piezo_pos_2 = get_frame_pos(frame_2)

    # Top is the first recorded position minus
    # the number of plane spacings down that recording took place
    t_top = piezo_pos_1 - (plane - 1) * (piezo_pos_2 - piezo_pos_1)

    # Now do the same thing for z
    z_tree = ElementTree.parse(z_xml)

    seq = z_tree.getroot().find('Sequence')
    frame_1 = seq.findall('Frame')[1]
    frame_2 = seq.findall('Frame')[2]

    plane = int(frame_1.get('index'))

    piezo_pos_1 = get_frame_pos(frame_1)
    piezo_pos_2 = get_frame_pos(frame_2)

    z_top = piezo_pos_1 - (plane - 1) * (piezo_pos_2 - piezo_pos_1)

    return t_top - z_top


def match_plane_to_z(t_plane, volume):
    corrs = []
    for z_plane in volume:
        corrs.append(np.corrcoef(
            (t_plane - np.mean(t_plane)).flatten(),
            (z_plane - np.mean(z_plane)).flatten())[0, 1])
    return np.nanargmax(corrs)


def directory_pairs_to_process(root_directory):

    dirs = [os.path.split(d)[0] for d in locate('*Cycle0*.h5', root_directory)]

    if len(dirs) == 1:
        # If a single file is given, generate pair by
        # altering name agnostic to whether T or Z path was given
        t_dirs = ['T'.join(dirs[0].rsplit('Z', 1))]
        z_dirs = ['Z'.join(dirs[0].rsplit('T', 1))]
    else:
        t_dirs = [d for d in dirs if os.path.basename(d).startswith('T')]
        z_dirs = [d for d in dirs if os.path.basename(d).startswith('Z')]

    # find adjacent T and Z series pairs
    pairs_to_process = []

    for t_dir in t_dirs:
        for z_dir in z_dirs:
            # directories should be exactly the same but with last Z a T
            if t_dir == 'T'.join(z_dir.rsplit('Z', 1)):
                pairs_to_process.append([t_dir, z_dir])

    pairs_to_process_sima = []
    for pair in pairs_to_process:
        [T_vol_path, Z_vol_path] = pair

        t_basename = os.path.basename(T_vol_path)
        z_basename = os.path.basename(Z_vol_path)

        pairs_to_process_sima.append(
            [join(T_vol_path, t_basename + '.sima'),
             join(Z_vol_path, z_basename + '.sima')])

    return pairs_to_process_sima


def volume_pattern_pairs_to_process(volume, functional_pattern):
    # Pairs directories matching functional pattern to volume path.
    functional_sima_folders = glob(functional_pattern)
    return [(f, volume) for f in functional_sima_folders]


class Branch(object):

    """Structure used to store dendritic branch morphological info

    Parameters
    ----------
    coords : 2D np.array of shape n x 3
        columns correspond to the x, y, and z coordinates of the branch points.
        coords are specified in pixel units
    parent : a Branch instance or None
        if parent is None, branch is assumed primary
    pixel_size : tuple of float, length 3
        The size in micron of a pixel in the x, y, and z dimensions

    """

    def __init__(self, coords, parent=None, pixel_size=(1., 1., 1.)):
        self.coords = coords
        self.parent = parent
        self.pixel_size = pixel_size
        self.children = []

    def __repr__(self):
        return '<Branch: id={id}, order={order}, length={length}>'.format(
            id=self.id, order=self.order, length=self.length())

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p
        if p is not None:
            p._add_child(self)

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, s):
        self._pixel_size = tuple([float(r) for r in s])

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, s):
        self._id = str(s)

    def _add_child(self, b):
        assert(np.all(self.coords[-1] == b.coords[0]))
        # assert(len(self.children) < 2)
        self.children.append(b)

    @property
    def order(self):
        if self._parent is None:
            return 1
        else:
            return self.parent.order + 1

    def length(self):
        # in um
        if not hasattr(self, '_length'):
            self._length = np.sum(
                np.linalg.norm(
                    (self.coords[1:] - self.coords[:-1]) * self.pixel_size,
                    axis=1))
        return self._length

    def distance_to_soma(self):
        # Not inclusive of self
        distance = 0
        parent = self.parent
        while parent is not None:
            distance += parent.length()
            parent = parent.parent
        return distance


def parse_swc(swc_file, primary_label='0'):
    """There is one .swc file per primary arbor.  Must provide the prefix
    label explicitly if a cell has multiple primary arbors
    """

    Z_SPACING = 2  # distance in micron between z-planes

    par_dir = dirname(dirname(dirname(dirname(swc_file))))
    xml_path = join(par_dir, os.path.basename(par_dir) + '.xml')
    pv_version = get_prairieview_version(xml_path)
    [y_spacing, x_spacing] = get_element_size_um(xml_path, pv_version)[-2:]
    pixel_size = (x_spacing, y_spacing, Z_SPACING)

    with open(swc_file, 'rb') as f:
        lines = f.readlines()[1:]

    data = []
    for l in lines:
        d = l.rstrip().split()
        # idx, x, y, z, parent
        data.append(
            [int(d[0]), float(d[2]), float(d[3]), float(d[4]), int(d[6])])
    data = np.array(data, dtype=float)

    # find branch point ids, sorted
    s = np.sort(data[:, -1], axis=None)
    match_idx = np.where(s[1:] == s[:-1])
    branch_ids = s[match_idx]
    terminal_ids = data[np.where(
        [x not in data[:, -1] for x in data[:, 0]])[0], 0]

    # .swc file contains only a single branch
    if not len(branch_ids):
        branch = Branch(data[:, 1:4], pixel_size=pixel_size)
        branch.id = primary_label
        return [branch]

    first_branch_id = branch_ids[0]
    first_branch_idx = np.argwhere(data[:, 0] == first_branch_id)[0][0]

    # inclusive of branch point
    branches = []
    primary_dendrite = Branch(
        data[:first_branch_idx + 1, 1:4], pixel_size=pixel_size)
    branches.append(primary_dendrite)

    dividing_branches = [primary_dendrite]
    dividing_branch_ids = [data[first_branch_idx, 0]]

    while len(dividing_branches):
        next_gen_branches = []
        next_gen_inds = []
        for parent_branch, branch_id in zip(
                dividing_branches, dividing_branch_ids):

            # find branches that start at the terminal point
            child_start_idxs = np.argwhere(data[:, -1] == branch_id)
            child_start_idxs = [x[0] for x in child_start_idxs]
            if len(child_start_idxs) == 0:
                # the principal dendrite was terminal
                pass
            else:
                # assert(len(child_start_idxs) == 2)
                for child_start_idx in child_start_idxs:
                    for c, dd in enumerate(data[child_start_idx + 1:, :]):
                        if dd[0] in branch_ids:
                            end_idx = child_start_idx + c + 1
                            terminal = False
                            break
                        if dd[0] in terminal_ids:
                            end_idx = child_start_idx + c + 1
                            terminal = True
                            break

                    # x, y, z
                    child_coords = np.vstack([data[np.argwhere(
                        data[:, 0] == branch_id)[0][0], 1:4],
                        data[child_start_idx: end_idx + 1, 1: 4]])

                    child_branch = Branch(
                        child_coords, parent_branch, pixel_size=pixel_size)
                    branches.append(child_branch)
                    if not terminal:
                        next_gen_inds.append(data[end_idx, 0])
                        next_gen_branches.append(child_branch)

        dividing_branches = next_gen_branches
        dividing_branch_ids = next_gen_inds

    primary_branch = [b for b in branches if b.order == 1][0]
    primary_branch.id = primary_label

    order = 2
    branches_of_interest = [b for b in branches if b.order == order]
    while len(branches_of_interest):
        for idx, branch in enumerate(branches_of_interest):
            # For most efficient coding count number of siblings,
            # Children are numbered as inde modulo this number
            num_children = len([b for b in branches_of_interest
                                if b.parent.id == branch.parent.id])
            branch.id = branch.parent.id + '_' + str(idx % num_children)
        order += 1
        branches_of_interest = [b for b in branches if b.order == order]

    return branches
