"""Losonczy Lab data archive utilities.

These utilities can be used to archive and restore files to a remote host.

Initially designed to work with Amazon Web Services S3.

Working with the AWS S3 bucket requires two packages:
>>> pip install --user boto3 awscli

To get started, you need to setup an account with AWS IAM that grants you
access to the lab S3 bucket and load your authentication keys.

See:
https://boto3.readthedocs.io/en/latest/guide/quickstart.html#configuration

To archive a file:
>>> archive('/data/user/mouse1/TSeries1/TSeries1.h5')

To restore a file:
>>> restore('/data/user/mouse1/TSeries1/TSeries1.h5.archive')

"""
import os
import boto3
import time
import json
import hashlib
import base64
import binascii

S3_BUCKET = 'losonczylab.data.archive'

# The number of a days a restored file will remain before being deleted.
# Files restored from Glacier storage are kept in both Glacier and a copy in
# Reduced Redundancy storage, so for this many days we pay double for the data
# storage.
GLACIER_RETENTION_PERIOD = 7

# Load S3 modules
boto3.resource('s3')
TRANSFER_CONFIG = boto3.s3.transfer.TransferConfig(
    multipart_threshold=8388608,
    max_concurrency=10,
    multipart_chunksize=8388608,
    num_download_attempts=5,
    max_io_queue=100,
    io_chunksize=262144)


def archive(local_path, host='aws', **kwargs):
    """Archive a local file to a remote host.

    Parameters
    ----------
    local_path : str
        Path to the local file to upload.
    host : {'aws'}
        Remote host/service to upload file to.
    kwargs : optional
        Additional host-specific arguments.

    """
    print("[{}] Beginning upload: {}".format(
        convert_time(time.time()), local_path))
    if host == 'aws':
        remote_path = aws_archive(local_path, **kwargs)
    else:
        raise TypeError("[{}] Unrecognized host: {}".format(
            convert_time(time.time()), host))

    if remote_path:
        os.remove(local_path)
        print("[{}] Upload successful: {}".format(
            convert_time(time.time()), local_path))
    else:
        print("[{}] Upload failed: {}".format(
            convert_time(time.time()), local_path))


def restore(placeholder_file, restore_in_place=False, **kwargs):
    """Restore a file from a remote host.

    Parameters
    ----------
    placeholder_file : str
        Path to the placeholder file (.archive) that contains information on
        the remote location of the file to restore.
    restore_in_place : bool
        If True, ignore the stored local file path in the placeholder file and
        restore the file in the same location as the placeholder file.
    kwargs : optional
        Additional host-specific keyword arguments.

    """
    data = parse_placeholder(placeholder_file)
    local_restore_path = placeholder_file.rstrip('.archive') if \
        restore_in_place else data['local_path']
    print("[{}] Beginning restore: {}".format(
        convert_time(time.time()), local_restore_path))
    if data.get('host') == 'aws':
        local_path = aws_restore(
            remote_path=data['remote_path'],
            local_path=local_restore_path,
            bucket=data['aws_bucket'],
            checksum_md5=data['checksum_md5'],
            orig_stat=data['stat'],
            **kwargs)
    else:
        raise TypeError("[{}] Unrecognized host: {}".format(
            convert_time(time.time()), data.get('host')))

    if local_path:
        if 'Glacier restore' in local_path:
            print("[{}] {}".format(convert_time(time.time()), local_path))
        else:
            os.remove(placeholder_file)
            print("[{}] File restored: {}".format(
                convert_time(time.time()), local_path))
    else:
        print("[{}] Restore failed: {}".format(
            convert_time(time.time()), placeholder_file))


def aws_archive(local_path, bucket=S3_BUCKET, storage_class='STANDARD',
                transfer_config=TRANSFER_CONFIG):
    """Upload a file to S3 and write the placeholder file.

    Parameters
    ----------
    local_path : str
        Path to file, will be mirrored in S3.
    bucket : str
        Name of S3 bucket to upload file to.
    storage_class : {'STANDARD', 'STANDARD_IA', 'REDUCED_REDUNDANCY'}
        Initial storage class of file. Lifecycle rules on the bucket might
        change this.
    transfer_config : boto3.s3.transfer.TransferConfig, optional
        Transfer configuration objects which describes the parameters used
        by the transfer manager.

    """
    local_path = os.path.abspath(local_path)
    remote_path = local_path.lstrip('/')

    checksum_md5 = md5(local_path)

    extra_args = {'StorageClass': storage_class,
                  'Metadata': {
                      'checksum_md5': checksum_md5,
                      'local_path': local_path,
                      'timestamp': convert_time(time.time())
                  }
                  }

    aws_upload(
        local_path,
        remote_path,
        bucket,
        ExtraArgs=extra_args,
        Config=transfer_config)

    if aws_verify(local_path, remote_path, bucket, transfer_config):
        write_placeholder(
            local_path, remote_path, host='aws', aws_bucket=bucket,
            checksum_md5=checksum_md5)
        return remote_path
    else:
        aws_delete(remote_path, bucket)
        return None


def aws_restore(remote_path, local_path, bucket=S3_BUCKET, checksum_md5=None,
                transfer_config=TRANSFER_CONFIG, orig_stat=None):
    """Restore a file from AWS S3 to the local machine.

    Parameters
    ----------
    remote_path : str
        Remote path to file.
    local_path : str
        Local path to file.
    bucket : str
        Name of S3 bucket.
    checksum_md5 : str, optional
    transfer_config : boto3.s3.transfer.TransferConfig, optional
        Transfer configuration objects which describes the parameters used
        by the transfer manager.
    orig_stat : dict, optional
        Dictionary of file stat properties.  If passed in, will attempt to
        restore some of the values. Should at least include: 'mode',
        'uid', and 'gid'.

    """
    file_query = aws_query(remote_path, bucket)
    if file_query.get('StorageClass', 'STANDARD') != 'GLACIER' or \
            file_query.get('Restore', '').startswith(
                'ongoing-request="false"'):
        aws_download(
            remote_path=remote_path,
            local_path=local_path,
            bucket=bucket,
            Config=transfer_config)

        if checksum_md5 is None or checksum_md5 == md5(local_path):
            if orig_stat:
                restore_file_stat(local_path, orig_stat)
            aws_delete(remote_path, bucket)
            return local_path
        else:
            return None
    else:
        if file_query.get('Restore') is None:
            aws_glacier_restore(remote_path, bucket=bucket)
            return "Glacier restore initiated: {}".format(local_path)
        elif file_query.get('Restore') == 'ongoing-request="true"':
            return "Glacier restore in progress: {}".format(local_path)

    return None


def aws_glacier_restore(remote_path, bucket=S3_BUCKET,
                        retention_period=GLACIER_RETENTION_PERIOD):
    """Initiate a restore of a file storage in Glacier storage.

    See: https://boto3.readthedocs.io/en/latest/reference/services/s3.html#id26

    Parameters
    ----------
    remote_path : str
        Remote path to file.
    bucket : str
        Name of S3 bucket.
    retention_period : int
        The number of a days a restored file copy should exist before being
        deleted. The original copy in Glacier storage is never deleted by this
        action.

    """
    s3 = boto3.client('s3')
    s3.restore_object(
        Bucket=bucket, Key=remote_path,
        RestoreRequest={'Days': retention_period})


def aws_verify(local_path, remote_path, bucket=S3_BUCKET,
               transfer_config=TRANSFER_CONFIG):
    """Compare a locally calculated eTag with the remote eTag.

    Parameters
    ----------
    local_path : str
        Local path to file.
    remote_path : str
        Remote path to file.
    bucket : str
        Name of S3 bucket.
    transfer_config : boto3.s3.transfer.TransferConfig, optional
        Transfer configuration objects which describes the parameters used
        by the transfer manager.

    """
    local_etag = etag(
        local_path,
        upload_max_size=transfer_config.multipart_threshold,
        upload_part_size=transfer_config.multipart_chunksize)
    file_query = aws_query(remote_path, bucket)

    return file_query.get('ETag', '').strip('"') == local_etag


def aws_upload(local_path, remote_path, bucket=S3_BUCKET, **kwargs):
    """Upload a file to AWS S3.

    Parameters
    ----------
    local_path : str
        Local path to file.
    remote_path : str
        Remote path to file.
    bucket : str
        Name of S3 bucket.
    kwargs : optional
        Additional arguments to pass directly to boto3.s3.upload_file.

    """
    s3 = boto3.client('s3')
    s3.upload_file(
        Filename=local_path, Bucket=bucket, Key=remote_path, **kwargs)


def aws_download(remote_path, local_path, bucket=S3_BUCKET, **kwargs):
    """Download a file from AWS S3.

    Parameters
    ----------
    remote_path : str
        Remote path to file.
    local_path : str
        Local path to file.
    bucket : str
        Name of S3 bucket.
    kwargs : optional
        Additional arguments to pass directly to boto3.s3.download_file.
    """
    s3 = boto3.client('s3')
    s3.download_file(
        Bucket=bucket, Key=remote_path, Filename=local_path, **kwargs)


def aws_delete(remote_path, bucket=S3_BUCKET):
    """Delete a file from AWS S3.

    Parameters
    ----------
    remote_path : str
        Remote path to file.
    bucket : str
        Name of S3 bucket.

    """
    s3 = boto3.client('s3')
    s3.delete_object(Bucket=bucket, Key=remote_path)


def aws_query(remote_path, bucket=S3_BUCKET):
    """Return the metadata associated with a remote file in AWS S3.

    Parameters
    ----------
    remote_path : str
        Remote path to file.
    bucket : str
        Name of S3 bucket.

    """
    s3 = boto3.client('s3')
    return s3.head_object(Bucket=bucket, Key=remote_path)


def write_placeholder(
        local_path, remote_path, host, checksum_md5=None,
        **additional_metadata):
    """Write placeholder file that references remote location.

    The placeholder file should contain all the information to locate the
    remote file and also verify that a file downloaded from the remote
    location matches the original file.

    Parameters
    ----------
    local_path : str
        Local path to file.
    remote_path : str
        Remote path to file.
    host : str
        Remote host/service where file was uploaded.
    checksum_md5 : str, optional
        MD5 checksum of the local file. If None, it will be calculated.
    additional_metadata : optional
        Additional host-specific information to store in the file.

    """
    placeholder_path = local_path + '.archive'
    if os.path.exists(placeholder_path):
        raise ValueError('File already exists: {}'.format(placeholder_path))

    st_mode, st_ino, st_dev, st_nlink, st_uid, st_gid, st_size, st_atime, \
        st_mtime, st_ctime = os.stat(local_path)

    if checksum_md5 is None:
        checksum_md5 = md5(local_path)

    data = {
        'stat': {
            'mode': st_mode,
            'ino': st_ino,
            'dev': st_dev,
            'nlink': st_nlink,
            'uid': st_uid,
            'gid': st_gid,
            'size': st_size,
            'atime': convert_time(st_atime),
            'mtime': convert_time(st_mtime),
            'ctime': convert_time(st_ctime)},
        'timestamp': convert_time(time.time()),
        'host': host,
        'local_path': local_path,
        'remote_path': remote_path,
        'checksum_md5': checksum_md5,
    }

    data.update(additional_metadata)

    json.dump(data, open(placeholder_path, 'w'), sort_keys=True, indent=4,
              separators=(',', ': '))


def parse_placeholder(placeholder_path):
    """Returned the parsed contents of a placeholder file."""
    return json.load(open(placeholder_path, 'r'))


def restore_file_stat(local_path, stat):
    """Attempt to restore the file properties to original values.

    Parameters
    ----------
    local_path : str
        Local path to file.
    stat : dict
        Dictionary of file stat properties. Should at least include: 'mode',
        'uid', and 'gid'.

    """
    try:
        os.chmod(local_path, stat['mode'])
    except OSError:
        pass
    try:
        os.chown(local_path, stat['uid'], stat['gid'])
    except OSError:
        pass


def md5(file_path):
    """Iteratively calculate the MD5 hash of a file.

    Should be equivalent to the shell command:
    >>> openssl md5 -binary file_path | base64

    Parameters
    ----------
    file_path : str
        Path to file.

    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest())


def etag(file_path, upload_max_size=TRANSFER_CONFIG.multipart_threshold,
         upload_part_size=TRANSFER_CONFIG.multipart_chunksize):
    """Calculate the same eTag that AWS will calculate after upload.

    The algorithm is different for multi-part uploads, so it depends on the
    size of the file.

    This is not officially supported by Amazon, so it could change in the
    future.

    Modified from:
    http://stackoverflow.com/questions/6591047/etag-definition-changed-in-amazon-s3

    Parameters
    ----------
    file_path : str
        Path to file.
    upload_max_size : int
        Max size of a file (in bytes) that will be uploaded as a single chunk.
    upload_part_size : int
        Size (in byes) of each chunk of a multi-part upload.

    """
    filesize = os.path.getsize(file_path)
    file_hash = hashlib.md5()

    if filesize > upload_max_size:

        block_count = 0
        md5string = ""
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(upload_part_size), ""):
                file_hash = hashlib.md5()
                file_hash.update(block)
                md5string = md5string + binascii.unhexlify(
                    file_hash.hexdigest())
                block_count += 1

        file_hash = hashlib.md5()
        file_hash.update(md5string)
        return file_hash.hexdigest() + "-" + str(block_count)

    else:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(upload_part_size), ""):
                file_hash.update(block)
        return file_hash.hexdigest()


def convert_time(time_in_seconds):
    return time.strftime(
        '%Y-%m-%d-%Hh%Mm%Ss', time.localtime(time_in_seconds))
