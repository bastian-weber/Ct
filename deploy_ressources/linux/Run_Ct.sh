#!/bin/sh

DIR="$(dirname "$(readlink -f "$0")")"
cd "$DIR"
LD_LIBRARY_PATH=LD_LIBRARY_PATH:. ./Ct "$@"

