#!/bin/bash

# Disable a USB driver to avoid warnings when importing OpenCV
ln /dev/null /dev/raw1394

exec "$@"
