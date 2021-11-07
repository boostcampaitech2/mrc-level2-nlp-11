#!/bin/bash

DIR="elasticsearch-7.9.2"
ZIP="elasticsearch-7.9.2-linux-x86_64.tar.gz"

pip install elasticsearch
if [ ! -e $DIR ]
then
    if [ -e $ZIP ]
    then
        tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
        chown -R daemon:daemon elasticsearch-7.9.2
        elasticsearch-7.9.2/bin/elasticsearch-plugin install analysis-nori
    else
        wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
        tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
        chown -R daemon:daemon elasticsearch-7.9.2
        elasticsearch-7.9.2/bin/elasticsearch-plugin install analysis-nori
    fi
fi

cp /opt/ml/mrc-level2-nlp-11/etc/my_stop_dic.txt ./elasticsearch-7.9.2/config/