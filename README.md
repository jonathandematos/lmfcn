# lmfcn
Large Margin Fully Convolutional Network

Sample of features extracted from Gaussian images being trained in 20 epochs:

https://user-images.githubusercontent.com/22577207/165535633-0ed95a0c-8f63-4348-8182-47e1fe491a32.mp4

Tested with:
- pytorch (1.11.0)
- py (3.9)
- cuda (11.3)
- cudnn (8.2)
- torchvision (0.12)
- scikit-learn (1.0.2)

Example of use with generated images:

`python3 main.py -i gaussians --epoch 5 --width 350 --height 230 --norm 0 --relax 1 --svclose 5 --shclose 0 --wrclose 1`

Example of use with images:

- First we need to create 3 files that is expected by the implementation, these files are `train.txt`,`test.txt` and `val.txt`
- each file follows the structure as follows:
  - each line from the txt file should have 2 elements splited by ";", the first element is the class 0 or 1, the second is the path to the image, e.g:
```
0;C:\path\to\my\instance\003af089-2533-4612-8a10-db178eac69d1_0.png
1;C:\path\to\my\instance\007725fe-0ec9-4863-92d6-68cba6f71e48_1.png
1;C:\path\to\my\instance\009c8747-86b8-4683-b41a-6aa4a0b5dc3c_1.png
0;C:\path\to\my\instance\00a6a3fe-dc7f-437e-8f3e-71ffe97839e0_0.png
1;C:\path\to\my\instance\00b0fc78-e308-492a-8a09-f5545c55a245_1.png
0;C:\path\to\my\instance\00d3b389-6090-4ef3-8305-1b1e41736a75_0.png
```

With the txt files ready, we can call the `main.py` with `--i` argument referencing our `train.txt`, the code will handle the other files for us:

`python main.py -i ".\train.txt" --epoch 5 --width 350 --height 230 --norm 0 --relax 1 --svclose 5 --shclose 0 --wrclose 1`
```
