#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
LD_LIBRARY_PATH=LD_LIBRARY_PATH:. ./CtViewer

