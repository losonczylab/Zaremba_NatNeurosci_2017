import sys

class ProgressBar():
    """ Simple command line progress bar class

    Example
    -------
    >>> from lab.misc.progressbar import ProgressBar
    >>> import time
    >>> num_iterations = 100
    >>> progress = ProgressBar(num_iterations)
    >>> for i in xrange(num_iterations)
    >>>     progress.update(i)
    >>>     time.sleep(1)
    >>> progress.end()


    Parameters
    ----------
    max_val : int
        the number of cycles equivalent to 100% progress
    display_length : int, optional
        the length, in number of characters, to display the progress bar
    marger : char, optional
        the character to use to represent progress
    """

    def __init__(self, max_val, display_length=20, marker='='):
        self._display_length = display_length
        self._max_val = max_val
        self._marker = marker

    def update(self, i):
        """Update the progress bar. Overwrites the current line in order to appear
        as moving accross

        Parameters
        ----------
        i : int
            Ther current cycle number/iteration to represent
        """

        sys.stdout.write('\r')
        per_comp=i*1.0/self._max_val
        text_str = "\t[%-{}s] %d%%".format(self._display_length)
        sys.stdout.write(text_str % (self._marker*int(per_comp*self._display_length),
            per_comp*100))
        sys.stdout.flush()

    def end(self):
        """End the progress bar - set to 100% and print a new line character """

        self.update(self._max_val)
        print ''
