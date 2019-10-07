#!/bin/bash
#
# Copyright 2019 Google LLC, Thea Sommerschield, Jonathan Prag
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo 'Regenerating dockerfile with UID...'
mv Dockerfile Dockerfile.old
( echo '### DO NOT EDIT DIRECTLY, SEE Dockerfile.template ###'; sed -e "s/<<UID>>/$UID/" < Dockerfile.template ) > Dockerfile
docker build -t $USER/fill .
