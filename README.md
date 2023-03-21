# DR-Pose
DRPose is developed on Ubuntu 18.04 OS, 3090 Ti GPU (CUDA 11.3).
To reproduce our results:

1. Create conda environment
```bash
conda create -n DRPose python=3.6
conda activate DRPose
```

2. Install basic packages
```bash
cd DRPose
pip install -r requirements.txt
```

3. install PyTorch + CUDA
```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

4. By running the following shell script, the other needed packages can be installed/compiled.
```bash
bash installer.sh
```

5. Download weights and extract under /DRPose/weights folder. The architecture should be:
```
|DRPose
 -|weights
 ---|registration
 ---|deformation
 ---|completion
```

6. Download dataset and extract to /DRPose/dataset folder. The architecture should be:
```
|DRPose
 -|dataset
 ---|NOCS
 -----|CAMERA
 -----|gts
 -----|obj_models
 -----|Real
 -----|results
```

7. For evaluaiton on CAMERA25 dataset:
```bash
python evaluate.py --dataset=camera_val
```

8. For evaluation on REAL275 dataset:
```bash
python evaluate.py --dataset=real_test
```
