#!/bin/bash

set -ex

# Build protoc
if test ! -e src/protoc; then
  ./autogen.sh
  ./configure
  make -j4
fi

umask 0022
pushd ruby
gem install bundler -v 2.1.4
bundle update && bundle exec rake gem:native
ls pkg
mv pkg/* $ARTIFACT_DIR
popd
