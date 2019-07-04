@echo off
call conda create -y -q -n dlnd19cifarcnns
call conda activate dlnd19cifarcnns

call conda install -y -q pytorch torchvision cudatoolkit=9.0 -c pytorch
call conda install -y -q jupyterlab numpy matplotlib

