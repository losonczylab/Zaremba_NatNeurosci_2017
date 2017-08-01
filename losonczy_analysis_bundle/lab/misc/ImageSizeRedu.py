# Resizing the tiff image
# Dont' forget the open the X11 ssh -X ...
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sima


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def full_range(im, max_val):
    """Let image use the full range 16bit range"""
    return (im.astype(float) / max_val * (2**16 - 1)).astype('uint16')


def top_bits(im, nbits):
    if im.dtype == 'uint16':
        return im & (2 ** 16 - 2 ** (16 - nbits))
    else:
        raise ValueError


def top_bits_round(im, nbits):
    return np.rint(im.astype(float) / (2 ** nbits)).astype(int) * (2 ** nbits)


def bit_corr(im1, im2, bit):
    return np.corrcoef(
        np.reshape(im1 & (2 ** (16 - bit)), -1),
        np.reshape(im2 & (2 ** (16 - bit)), -1))


def fold_im(im, nbits):
    """Fold an image with maximum value 2**nbits"""
    im = im.astype(float)
    bottom = im * (im < 2 ** (nbits-1))
    top = (im >= 2**(nbits-1)) * (2**nbits - 1 - im)
    return (bottom + top).astype('uint16')


def analyze_channel(dataset):
    """Assumes a one channel dataset"""
    max_val = - np.inf
    for im in itertools.chain.from_iterable(dataset):
        max_val = max(max_val, np.nanmax(im))
    im_counter = 0
    error = np.zeros(16)
    corrcoef_sum = np.zeros(16)
    fold_corr_sum = np.zeros(16)
    count = 0
    for im1, im2 in pairwise(itertools.chain.from_iterable(dataset)):
        im1 = im1.astype('uint16')  # full_range(im1, max_val)
        im2 = im2.astype('uint16')  # full_range(im2, max_val)
        fold1 = im1.copy()
        fold2 = im2.copy()
        for nbits in range(1, 17):
            error[nbits-1] += np.sum(
                np.abs(top_bits(im2, nbits).astype(float) - im1.astype(float)))
            corrcoef_sum[nbits-1] += bit_corr(im1, im2, nbits)[0, 1]
            fold_corr_sum[nbits-1] += bit_corr(fold1, fold2, nbits)[0, 1]
            fold1 = fold_im(fold1, 17 - nbits)
            fold2 = fold_im(fold2, 17 - nbits)

        count += im1.size
        im_counter += 1
        if im_counter > 10:
            break
    return error / count, corrcoef_sum / im_counter, fold_corr_sum / im_counter


def analyze(dataset):
    errors, corrs, fold_corrs = [], [], []
    for channel in range(dataset.frame_shape[3]):
        res = analyze_channel(dataset[:, :, :, :, :, channel])
        errors.append(res[0])
        corrs.append(res[1])
        fold_corrs.append(res[2])

    fig = plt.figure()
    axes = [fig.add_subplot(311),
            fig.add_subplot(312),
            fig.add_subplot(313)]
    for chan in zip(errors, corrs, fold_corrs):
        for i in range(3):
            axes[i].plot(range(1, 17), chan[i])
    axes[0].set_ylabel('error')
    axes[0].legend([str(c+1) for c in range(len(errors))])
    axes[1].set_ylabel('corr')
    axes[2].set_ylabel('fold corr')
    for i in range(1, 3):
        axes[i].plot([1, 16], [0, 0], 'k')
    plt.show()


if __name__ == '__main__':
    path = '/data/Joseph/Imaging/JT39/2014-09-13/TSeries-FOV1-000/TSeries-FOV1-000_Cycle00001_Element00001.h5'
    seq = sima.Sequence.create('HDF5', path, 'tzyxc')[5:10, ::20]
    ds = sima.ImagingDataset([seq], savedir=None)

    # path = '/data/Jeff/running/jz107/TSeries-12112014-Day1-Session1-000/TSeries-12112014-Day1-Session1-000_Cycle00001_Element00001.h5'
    # path = '/data/Nathan/2photon/acuteRemapping/nd99/10282014/ctxA-002/ctxA-002_Cycle00001_Element00001.h5'
    # path = '/data/Nathan/2photon/acuteRemapping/nd113/12102014/ctxA-002/ctxA-002_Cycle00001_Element00001.h5'
    # path = '/data/Nathan/2photon/acuteRemapping/nd101/10292014/ctxB-001/ctxB-001_Cycle00001_Element00001.h5'
    seq = sima.Sequence.create('HDF5', path, 'tzyxc')[0:20]

    ds = sima.ImagingDataset([seq], savedir=None)
    print "Analyze:"
    analyze(ds)
