# lmfcn
Large Margin Fully Convolutional Network

<video src='out.mp4' width=180></video>

Tested with:
- pytorch (1.11.0)
- py (3.9)
- cuda (11.3)
- cudnn (8.2)
- torchvision (0.12)
- scikit-learn (1.0.2)

Example of use:
python3 main.py -i gaussians --epoch 5 --width 350 --height 230 --norm 0 --relax 1 --svclose 5 --shclose 0 --wrclose 1
