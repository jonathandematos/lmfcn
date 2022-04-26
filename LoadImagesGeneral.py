import numpy as np
from PIL import Image
import PIL
from tqdm import tqdm
from os.path import isdir, isfile

import random

from torchvision import transforms

import torch

class LoadImages():
    def __init__(self, imglist=None, X=None, y=None, width=200, height=200, batch_size=16, randomize=False, categorical=False, x_min=None, x_max=None, aug=1, classcount=None, mislabel=0.0):
        self.X = list()
        self.y = list()

        x_min = None
        x_max = None

        self.width = width
        self.height = height
        self.channels = 3
        self.idx_count = 0
        self.dump = False
        self.batch_size = batch_size
        self.randomize = randomize
        self.classcount = list()
        self.classes = list()
        self.categorical = categorical
        self.aug = aug
        self.X_batches = None
        self.y_batches = None
        self.names = list()
        self.mislabel = mislabel
        augmentation = None

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406],
                #    std=[0.229, 0.224, 0.225]
                #)
            ])

        self.size = 0

        #self.load_directory(imglist)

        if(not imglist is None):
            with open(imglist) as f:
                for i in f:
                    self.size +=1
        else:
            self.size = X.shape[0]

        #############################

        self.orgsize = self.size
        self.augsize = self.size * self.aug

        if(not augmentation is None):
            self.aug_types = augmentation
        else:
            #self.aug_types = {'bright':0.25, 'contrast':0.25, 'saturation':0.25, 'hue':0.04, 'rotation':[-10,10], 'hflip':0.5, 'vflip':0.5}
            self.aug_types = {'bright':0.25, 'contrast':0.25, 'saturation':0.25, 'hue':0.04, 'hflip':0.5, 'vflip':0.5}
            #self.aug_types = {'bright':0.1, 'contrast':0.1, 'saturation':0.1, 'hue':0.02, 'hflip':0.5, 'vflip':0.5}
            #self.aug_types = {'bright':0.1, 'contrast':0.1, 'saturation':0.1, 'hue':0.02}
            #self.aug_types = {'hflip':0.5, 'vflip':0.5, 'rotation':[-15,15]}
            #self.aug_types = {'rotation':[-15,15]}

        self.augs = dict()
        self.innitalize_augmentation()

        ############################
        
        if(X is None and not imglist is None):
            pbar = tqdm(total=self.size * self.aug, ncols=79, disable=False)
            for k in range(self.aug):
                with open(imglist) as f:
                    for i in f:
                        filename = i.split(";")[-1][:-1]
                        self.names.append(filename.split("/")[-1])
                        image = Image.open(filename).convert("RGB")
                        image = image.resize((self.width,self.height), PIL.Image.BILINEAR)
                        image = np.array(image).astype("float32")
                        image /= 255.0
                        #image = ( image - image.min() ) / ( image.max() - image.min() )
                        #image -= 0.5
                        #image *= 2

                        self.X.append(image)
                        lbl = int(i.split(";")[0])
                        self.y.append(lbl)
                        if(lbl not in self.classes):
                            self.classes.append(lbl)
                        pbar.update()
            pbar.close()
            self.X = np.array(self.X)
            self.X = np.moveaxis(self.X, -1, 1)
            self.y = np.array(self.y)
        else:
            self.X = X
            self.y = y


        if(x_min is None or x_max is None):
            self.x_min = self.X.min()
            self.x_max = self.X.max()
        else:
            self.x_min = x_min
            self.x_max = x_max
        
        #self.X = (self.X - self.x_min) / (self.x_max - self.x_min)

        if(self.mislabel > 0.0):
            self.mislabeling()

        #print(self.x_max, self.x_min)

        self.classcount = [0 for i in range(len(self.classes))]

        for i in range(len(self.classes)):
            self.classcount[i] = np.where(self.classes[i] == self.y)[0].shape[0]

        if(self.dump == True):
            #im = Image.fromarray(np.uint8(gauss))
            #im.save("dump/{:03d}.png".format(self.idx_count))
            #self.idx_count += 1
            pass

        self.batch_indexes = list()

        batch_rand_index = np.arange(self.X.shape[0])
        if(self.randomize == True):
            np.random.shuffle(batch_rand_index)

        self.nr_batches = self.X.shape[0] // self.batch_size
        batch_samples = self.nr_batches * self.batch_size
        batch_rest = self.X.shape[0] - batch_samples

        for i in range(0,self.nr_batches):
            element = i * self.batch_size
            self.batch_indexes.append(batch_rand_index[element:element + self.batch_size])

        if(self.X.shape[0] % self.batch_size > 0):
            element = self.nr_batches * self.batch_size
            self.batch_indexes.append(batch_rand_index[element:element + batch_rest])
            self.nr_batches += 1

    ############################# methods #############################    

    def load_directory(self, directory):
        if(isdir(directory)):
            files = [f for f in listdir(directory) if isfile(join(directory, f))]
            print(files)

    def mislabeling(self):
        size = self.y.shape[0]

        change = np.arange(0,size,1)

        np.random.shuffle(change)

        nb_changes = int(self.y.shape[0]*self.mislabel)

        for i in change[:nb_changes]:
            tmp = self.classes.copy()
            tmp.remove(self.y[i])

            random.shuffle(tmp)

            print("Changed from {} to {}".format(self.y[i], tmp[0]))

            self.y[i] = tmp[0]

    def innitalize_augmentation(self):
        aug = self.aug-1
        if(aug > 0):
            if("bright" in self.aug_types.keys()):
                self.augs["bright"] = 1+((np.random.rand((self.size*aug))-0.5)*2*self.aug_types["bright"])
            if("contrast" in self.aug_types.keys()):
                self.augs["contrast"] = 1+((np.random.rand((self.size*aug))-0.5)*2*self.aug_types["contrast"])
            if("saturation" in self.aug_types.keys()):
                self.augs["saturation"] = 1+((np.random.rand((self.size*aug))-0.5)*2*self.aug_types["saturation"])
            if("hue" in self.aug_types.keys()):
                self.augs["hue"] = (np.random.rand((self.size*aug))-0.5)*self.aug_types["hue"]
            if("rotation" in self.aug_types.keys()):
                self.augs["rotation"] = np.random.randint(low=self.aug_types["rotation"][0], high=self.aug_types["rotation"][1], size=((self.size*aug)))
            if("hflip" in self.aug_types.keys()):
                self.augs["hflip"] = np.random.rand((self.size*aug)) < self.aug_types["hflip"]
            if("vflip" in self.aug_types.keys()):
                self.augs["vflip"] = np.random.rand((self.size*aug)) < self.aug_types["vflip"]

    def apply_augmentation(self, img, idx):
        q = transforms.ToPILImage()
        img = q(img)
        for i in self.aug_types.keys():
            if(i == "bright"):
                img = transforms.functional.adjust_brightness(img, self.augs["bright"][idx])
            if(i == "contrast"):
                img = transforms.functional.adjust_contrast(img, self.augs["contrast"][idx])
            if(i == "saturation"):
                img = transforms.functional.adjust_saturation(img, self.augs["saturation"][idx])
            if(i == "hue"):
                img = transforms.functional.adjust_hue(img, self.augs["hue"][idx])
            if(i == "rotation"):
                img = transforms.functional.rotate(img, self.augs["rotation"][idx])
            if(i == "hflip"):
                if(self.augs["hflip"][idx]):
                    img = transforms.functional.hflip(img)
            if(i == "vflip"):
                if(self.augs["vflip"][idx]):
                    img = transforms.functional.vflip(img)
        return img
 
    def get_tensor(self, idx=0):
        if(not self.X_batches is None):
            return self.X_batches[idx].reshape(1,self.channels, self.height, self.width), self.y[idx]
        else:
            #im = Image.fromarray(np.uint8(self.X[idx]))

            tt = transforms.ToTensor()

            #im_tmp = self.X[idx]
            #im_tmp = (self.X[idx] - self.x_min) / (self.x_max - self.x_min)
            #im_tmp = (self.X[idx] - self.X.min()) / (self.X.max() - self.X.min())

            #print(im_tmp.shape)
            
            if(idx >= self.size):
                im = tt(self.apply_augmentation(torch.tensor(self.X[idx]), idx - self.size)).reshape(1,self.channels, self.height, self.width)
            else:
                im = torch.tensor(self.X[idx].reshape((1,self.channels, self.width, self.height)))

            return im, self.y[idx]
 
    def get_tensors(self):
        X_tensor = None
       
        pbar = tqdm(total=self.size * self.aug, ncols=79, disable=True)

        X_tensor = torch.tensor(self.X)
        pbar.update(self.size)

        to_tensor = transforms.ToTensor()

        for i in range(self.size, self.X.shape[0]):
            X_tensor[i] = to_tensor(self.apply_augmentation(X_tensor[i], i - self.size))
            pbar.update()
        pbar.close()

        return X_tensor, self.y

    def generate_batches(self):
        self.X_batches, self.y_batches = self.get_tensors()
        del self.X

    def get_batch(self, idx=0):
        if(not self.X_batches is None):
            if(self.categorical):
                y_array = self.to_categorical(self.y_batches[self.batch_indexes[idx]])
                return self.X_batches[self.batch_indexes[idx]], y_array
            else:
                return self.X_batches[self.batch_indexes[idx]], self.y_batches[self.batch_indexes[idx]]
        else:
            y_array = np.zeros((self.batch_size), dtype=int)
            X_tmp = None
            tt = transforms.ToTensor()
            pbar = tqdm(total=self.batch_indexes[idx].shape[0], ncols=79, disable=True)
            for i in self.batch_indexes[idx]:
                if(i >= self.size):
                    if(X_tmp is None):
                        X_tmp = tt(self.apply_augmentation(torch.tensor(self.X[i]), i - self.size)).reshape(1,self.channels, self.height, self.width)
                    else:
                        X_tmp = torch.cat((X_tmp, tt(self.apply_augmentation(torch.tensor(self.X[i]), i - self.size)).reshape(1,self.channels, self.height, self.width)))
                else:
                    if(X_tmp is None):
                        X_tmp = torch.tensor(self.X[i]).reshape(1,self.channels, self.height, self.width)
                    else:
                        X_tmp = torch.cat((X_tmp, torch.tensor(self.X[i]).reshape(1,self.channels, self.height, self.width)))
                pbar.update()
            pbar.close()

            #X_tmp = (self.X[self.batch_indexes[idx]] - self.x_min) / (self.x_max - self.x_min) 
            #X_tmp = (self.X[self.batch_indexes[idx]] - self.X.min()) / (self.X.max() - self.X.min()) 
            y_array = self.y[self.batch_indexes[idx]]

            if(self.categorical):
                y_array = self.to_categorical(y_array)

            return X_tmp, y_array

    def to_categorical(self, y):
        yy = np.zeros((y.shape[0], self.y.max()+1))
        for i in range(y.shape[0]):
            yy[i][y[i]] = 1
        return yy
