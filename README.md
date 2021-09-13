# FibNet

**Development Environment**
Linux 20.04
Anaconda Virtual environment

**Build Requirements:**

pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install torchsummary

pip3 install matplotlib

pip3 install tqdm

pip3 install ptflops

**Run Training**

python main.py --arch FibNet  --ds ImageNet --n-blocks 5 --  --block-depth 3  --r1 0.68 --r2 0.24 --num_class 1000 --batch-size 256 --learning-rate 0.0005 --gpu 0
