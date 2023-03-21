#!/bin/bash


cd pointnet2/pointnet2
python setup.py install
cd ../..

cd lib/nn_distance
python setup.py install
cd ../..

cd cpp_wrappers
sh compile_wrappers.sh
cd ..


cd extensions
cd chamfer_dist
python setup.py install --user
cd ..

cd cubic_feature_sampling
python setup.py install --user
cd ..

cd gridding
python setup.py install --user
cd ..

cd gridding_loss
python setup.py install --user
cd ../..

cd PoinTr
pip install pointnet2_ops_lib/.

pip install pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl