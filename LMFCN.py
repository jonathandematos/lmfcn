import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from timeit import default_timer as timer
from datetime import timedelta
import torch
import torch.optim as optim
from TCNN import TCNN

PAUSE = False

# Controlled experimentation
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

cuda = torch.device('cuda')
cpu = torch.device('cpu')

class LMFCN:

    def __init__(self, feature=16, width=200, height=200, epoch=20, 
            gamma=1.0, relax=1000, lr=0.1, shake=0, shclose=1, svclose=1, 
            wrclose=0, aug=1, norm=1, seed=0, outdir="graphs/", outfile="fcnsv_output", batch=256, i=None,
            rho=0.65, eps=1e-6, wd=0.1):

        self.feature = feature
        self.width = width
        self.height = height
        self.epoch = epoch
        self.gamma = gamma
        self.relax = relax
        self.lr = lr
        self.shake = shake
        self.shclose = shclose
        self.svclose = svclose
        self.wrclose = wrclose
        self.aug = aug
        self.norm = norm
        self.seed = seed
        self.outfile = outfile
        self.outdir = outdir
        self.batch = batch
        self.i = i
        self.RHO=rho
        self.EPS=eps
        self.wd = wd
        self.timing = False

        NR_INST = 200

        self.model = TCNN(output_size=self.feature)
        self.model.to(cuda)

        #summary(self.model, (3,self.height,self.width))
       
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr, rho=self.RHO, eps=self.EPS, weight_decay=self.wd)
        #optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.5, weight_decay=self.wd)
        #optimizer = optim.Adadelta(model.parameters(), lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.wd)
        #optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=self.eps, weight_decay=self.wd, momentum=0, centered=False)
        #optimizer = optim.Adagrad(model.parameters(), lr=self.lr, lr_decay=0, weight_decay=self.wd, initial_accumulator_value=0)
        
        #scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.8, last_epoch=-1) #, verbose=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch, eta_min=1e-2, last_epoch=-1) #, verbose=False)

        self.svm = None

        self.li_tr = None
        self.li_ts = None
        self.li_val = None

        if(not i is None):
            if(i == "gaussians"):
                from LoadGaussians import LoadImages
            else:
                from LoadImagesGeneral import LoadImages

            ts = self.i.replace("train","test")
            tr_val = self.i.replace("train","val")
            tr = self.i
            #print("Data loading from: {}, {} and {}".format(tr, ts, tr_val))

            self.li_tr = LoadImages(imglist=tr, batch_size=self.batch, aug=self.aug, width=self.width, height=self.height, classcount=[NR_INST//2,NR_INST//2])
            self.li_ts = LoadImages(imglist=ts, batch_size=self.batch, aug=1, width=self.width, height=self.height, x_min=self.li_tr.x_min, x_max=self.li_tr.x_max, classcount=[NR_INST//2,NR_INST//2])
            self.li_val = LoadImages(imglist=tr_val, batch_size=self.batch, aug=1, width=self.width, height=self.height, x_min=self.li_tr.x_min, x_max=self.li_tr.x_max, classcount=[NR_INST//2,NR_INST//2])
        
            self.li_tr.generate_batches()
            self.li_ts.generate_batches()
            self.li_val.generate_batches()

    def predict(self, X=None, y=None):
        X_ts = None
        y_ts = None
        
        self.model.eval()
        torch.set_grad_enabled(False)
 
        if(X is not None):
            from LoadImagesGeneral import LoadImages

            self.li_ts = LoadImages(imglist=None, X=X, y=y, batch_size=32, aug=1, width=self.width, height=self.height)
            self.li_ts.generate_batches()
        else:
            if(self.li_ts is None):
                exit(0)

        for i in range(self.li_ts.nr_batches):
            X_input, y_input = self.li_ts.get_batch(idx=i)
            X_tmp = self.model(X_input.to(cuda))

            if(X_ts is None):
                X_ts = X_tmp
                y_ts = y_input
            else:
                X_ts = torch.cat((X_ts, X_tmp),0)
                y_ts = np.concatenate((y_ts, y_input),0)
                del X_input, y_input, X_tmp
        
        qsqrt_tensor_ts = torch.zeros((X_ts.shape[0],self.X.shape[0]))

        for i in range(qsqrt_tensor_ts.shape[0]):
            a = torch.dot(X_ts[i],X_ts[i])+torch.pow(self.X, 2).sum(1)-(X_ts[i]*self.X).sum(1)*2
            qsqrt_tensor_ts[i] = torch.exp(-self.gamma_auto * a)

        qsqrt_sq_ts = qsqrt_tensor_ts.detach().cpu().numpy()
        X_ts = X_ts.detach().cpu().numpy()
        y_true_ts, y_pred_ts = y_ts, self.clf.predict(qsqrt_sq_ts)
        class_report_ts = classification_report(y_true_ts, y_pred_ts, output_dict=True) #, zero_division=1)
        bal_acc_ts_reg = accuracy_score(y_true_ts, y_pred_ts)
        bal_acc_ts = balanced_accuracy_score(y_true_ts, y_pred_ts)
        print("SVM ts:  {:.4f} ({:.4f}) Recall(0): {:.4f} Recall(1): {:.4f}".format(
                bal_acc_ts,
                bal_acc_ts_reg,
                class_report_ts['0']['recall'],               
                class_report_ts['1']['recall']               
            ))

        return(y_pred_ts)
 
    def predict_proba(self, X=None, y=None):
        X_ts = None
        y_ts = None
        
        self.model.eval()
        torch.set_grad_enabled(False)
 
        if(X is not None):
            if(len(X.shape) == 3):
                X_ts = self.model(torch.tensor(X).to(cuda).unsqueeze(0))
            elif X.shape[0] == 1 and len(X.shape) == 4:
                X_ts = self.model(torch.tensor(X).to(cuda))
            else:
                from LoadImagesGeneral import LoadImages
    
                self.li_ts = LoadImages(imglist=None, X=X, y=y, batch_size=32, aug=1, width=self.width, height=self.height)
                self.li_ts.generate_batches()
                for i in range(self.li_ts.nr_batches):
                    X_input, y_input = self.li_ts.get_batch(idx=i)
                    X_tmp = self.model(X_input.to(cuda))

                    if(X_ts is None):
                        X_ts = X_tmp
                        y_ts = y_input
                    else:
                        X_ts = torch.cat((X_ts, X_tmp),0)
                        y_ts = np.concatenate((y_ts, y_input),0)
                        del X_input, y_input, X_tmp

        else:
            if(self.li_ts is None):
                exit(0)

            for i in range(self.li_ts.nr_batches):
                X_input, y_input = self.li_ts.get_batch(idx=i)
                X_tmp = self.model(X_input.to(cuda))

                if(X_ts is None):
                    X_ts = X_tmp
                    y_ts = y_input
                else:
                    X_ts = torch.cat((X_ts, X_tmp),0)
                    y_ts = np.concatenate((y_ts, y_input),0)
                    del X_input, y_input, X_tmp
        
        qsqrt_tensor_ts = torch.zeros((X_ts.shape[0],self.X.shape[0]))

        for i in range(qsqrt_tensor_ts.shape[0]):
            a = torch.dot(X_ts[i],X_ts[i])+torch.pow(self.X, 2).sum(1)-(X_ts[i]*self.X).sum(1)*2
            qsqrt_tensor_ts[i] = torch.exp(-self.gamma_auto * a)

        qsqrt_sq_ts = qsqrt_tensor_ts.detach().cpu().numpy()
        X_ts = X_ts.detach().cpu().numpy()
        y_true_ts, y_pred_ts = y_ts, self.clf.predict_proba(qsqrt_sq_ts)

        return y_pred_ts.squeeze()

    def fit(self, X=None, y=None, X_val=None, y_val=None, X_test=None, y_test=None):
        DEBUG = False
        GAMMA = self.gamma
        RELAX = self.relax
        LR = self.lr

        outputs = list()

        current_lr = 0.0
        max_val = 0.0
        max_tr = 0.0
        best_ts = 0.0
        best_tr = 0.0
        best_val = 0.0
        best_epoch = 0

        li_tr = self.li_tr
        li_ts = self.li_ts
        li_val = self.li_val

        if(not X is None):
            from LoadImagesGeneral import LoadImages

            li_tr = LoadImages(X=X, y=y, batch_size=self.batch, aug=self.aug, width=self.width, height=self.height)
            li_tr.generate_batches()
        if(not X_test is None):
            li_ts = LoadImages(X=X_test, y=y_test, batch_size=self.batch, aug=1, width=self.width, height=self.height, x_min=li_tr.x_min, x_max=li_tr.x_max)
            li_ts.generate_batches()
        if(not X_val is None):

            li_val = LoadImages(X=X_val, y=y_val, batch_size=self.batch, aug=1, width=self.width, height=self.height, x_min=li_tr.x_min, x_max=li_tr.x_max)
            li_val.generate_batches()

        list_bal_tr = list()
        list_bal_ts = list()
        list_bal_val = list()
        list_bal_tr_sp = list()
        list_bal_ts_sp = list()
        list_bal_val_sp = list()
        
        val_loss = list()
        val_loss_wrong = list()
        val_loss_opposite = list()
        nr_sups = list()

        for epoch in range(self.epoch):
        
            if(DEBUG == True):
                print("Forward pass")
        
            self.model.eval()
            torch.set_grad_enabled(False)
    
            if(self.timing):
                start = timer()
        
            ################# TRAIN SET #################
        
            X = None
            y_tr = None
        
            for i in range(li_tr.nr_batches):
                X_input, y_input = li_tr.get_batch(idx=i)
                X_tmp = self.model(X_input.to(cuda))
        
                if(X is None):
                    X = X_tmp
                    y_tr = y_input
                else:
                    X = torch.cat((X, X_tmp),0)
                    y_tr = np.concatenate((y_tr, y_input),0)
        
            del X_input, y_input, X_tmp
        
            if(self.timing):
                end = timer()
                extract = timedelta(seconds=end-start)
        
            GAMMA = 1 / (X.shape[1] * X.var()) * self.gamma
            self.gamma_auto = GAMMA
        
            qsqrt_tensor = torch.zeros((X.shape[0], X.shape[0]))
            qsqrt_dist = torch.zeros((X.shape[0], X.shape[0]))
        
            for i in range(qsqrt_tensor.shape[0]):
                a = torch.dot(X[i], X[i])+torch.pow(X, 2).sum(1)-(X[i]*X).sum(1)*2
                qsqrt_tensor[i] = torch.exp(-GAMMA * a)
                qsqrt_dist[i] = torch.sqrt(a)
        
            qsqrt_sq = qsqrt_tensor.detach().cpu().numpy()
        
            #if(self.timing):
            #    start = timer()
            ##qsqrt_dist = torch.cdist(X, X, p=2.0)
            ###qsqrt_dist1 = qsqrt_dist.detach().cpu().numpy()
            ###qsqrt_sq = np.power(qsqrt_dist1,2)
            ###qsqrt_sq = np.exp(qsqrt_sq)
            ##qsqrt_sq = torch.exp(-GAMMA*torch.pow(qsqrt_dist, 2)).detach().cpu().numpy()
        
            #if(self.timing):
            #    end = timer()
            #    distancetime = timedelta(seconds=end-start)
        
            clf = SVC(C=RELAX, kernel="precomputed", probability=True)
            self.clf = clf
            self.clf.fit(qsqrt_sq, y_tr)
            self.X = X

            if(self.timing):
                end2 = timer()
                smotime = timedelta(seconds=end2-end)
        
            ################# VAL SET #################
       
            if(li_val is not None):

                X_val = None
                y_val = None
            
                for i in range(li_val.nr_batches):
                    X_input, y_input = li_val.get_batch(idx=i)
                    X_tmp = self.model(X_input.to(cuda))
            
                    if(X_val is None):
                        X_val = X_tmp
                        y_val = y_input
                    else:
                        X_val = torch.cat((X_val, X_tmp),0)
                        y_val = np.concatenate((y_val, y_input),0)
            
                del X_input, y_input, X_tmp
            
                qsqrt_tensor_val = torch.zeros((X_val.shape[0],X.shape[0]))
            
                for i in range(qsqrt_tensor_val.shape[0]):
                    a = torch.dot(X_val[i],X_val[i])+torch.pow(X, 2).sum(1)-(X_val[i]*X).sum(1)*2
                    qsqrt_tensor_val[i] = torch.exp(-GAMMA * a)
        
            ################# TEST SET #################
        
            if(li_ts is not None):

                X_ts = None
                y_ts = None
            
                for i in range(li_ts.nr_batches):
                    X_input, y_input = li_ts.get_batch(idx=i)
                    X_tmp = self.model(X_input.to(cuda))
            
                    if(X_ts is None):
                        X_ts = X_tmp
                        y_ts = y_input
                    else:
                        X_ts = torch.cat((X_ts, X_tmp),0)
                        y_ts = np.concatenate((y_ts, y_input),0)
            
                del X_input, y_input, X_tmp
            
                qsqrt_tensor_ts = torch.zeros((X_ts.shape[0],X.shape[0]))
            
                for i in range(qsqrt_tensor_ts.shape[0]):
                    a = torch.dot(X_ts[i],X_ts[i])+torch.pow(X, 2).sum(1)-(X_ts[i]*X).sum(1)*2
                    qsqrt_tensor_ts[i] = torch.exp(-GAMMA * a)
        
            ################# PREDICTION REPORT #################

            print("------------------ Epoch {} --------------------".format(epoch))

            output_line = dict()

            qsqrt = qsqrt_dist.detach().cpu().numpy()
            X = X.detach().cpu().numpy()
            y_true, y_pred = y_tr, clf.predict(qsqrt_sq)
            class_report_tr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            bal_acc_tr_reg = accuracy_score(y_true, y_pred)
            bal_acc_tr = balanced_accuracy_score(y_true, y_pred)
            print("SVM tr:  {:.4f} ({:.4f}) Recall(0): {:.4f} Recall(1): {:.4f} Supports: {:4d}".format(
                    bal_acc_tr,
                    bal_acc_tr_reg,
                    class_report_tr['0']['recall'],               
                    class_report_tr['1']['recall'],               
                    len(clf.support_)
                ))
            
            output_line = {'train balanced acc':bal_acc_tr,
                    'train acc':bal_acc_tr_reg,
                    'recall 0':class_report_tr['0']['recall'],
                    'recall 1':class_report_tr['1']['recall'],
                    'svs':len(clf.support_)}

            if(not li_val is None):
                qsqrt_sq_val = qsqrt_tensor_val.detach().cpu().numpy()
                X_val = X_val.detach().cpu().numpy()
                y_true_val, y_pred_val = y_val, clf.predict(qsqrt_sq_val)
                class_report_val = classification_report(y_true_val, y_pred_val, output_dict=True, zero_division=0)
                bal_acc_val_reg = accuracy_score(y_true_val, y_pred_val)
                bal_acc_val = balanced_accuracy_score(y_true_val, y_pred_val)
                print("SVM val: {:.4f} ({:.4f}) Recall(0): {:.4f} Recall(1): {:.4f}".format(
                        bal_acc_val,
                        bal_acc_val_reg,
                        class_report_val['0']['recall'],               
                        class_report_val['1']['recall']               
                    ))
                output_line['val balanced acc'] = bal_acc_val
                output_line['val acc'] = bal_acc_val_reg
                output_line['val recall 0'] = class_report_val['0']['recall']
                output_line['val recall 1'] = class_report_val['1']['recall']


            if(not li_ts is None):
                qsqrt_sq_ts = qsqrt_tensor_ts.detach().cpu().numpy()
                X_ts = X_ts.detach().cpu().numpy()
                y_true_ts, y_pred_ts = y_ts, clf.predict(qsqrt_sq_ts)
                class_report_ts = classification_report(y_true_ts, y_pred_ts, output_dict=True, zero_division=0)
                bal_acc_ts_reg = accuracy_score(y_true_ts, y_pred_ts)
                bal_acc_ts = balanced_accuracy_score(y_true_ts, y_pred_ts)
                print("SVM ts:  {:.4f} ({:.4f}) Recall(0): {:.4f} Recall(1): {:.4f}".format(
                        bal_acc_ts,
                        bal_acc_ts_reg,
                        class_report_ts['0']['recall'],               
                        class_report_ts['1']['recall']               
                    ))
        
                output_line['ts balanced acc'] = bal_acc_ts
                output_line['ts acc'] = bal_acc_ts_reg
                output_line['ts recall 0'] = class_report_ts['0']['recall']
                output_line['ts recall 1'] = class_report_ts['1']['recall']

            outputs.append(output_line)

            for i in np.argwhere(np.isnan(qsqrt_sq)):
                print("Model error, try other hyperparameters")
                exit(0)
       
            ################# DETERMINE INTEREST AND CLOSEST ######################
        
            if(DEBUG == True):
                print("Calculate closest")
        
            wrong = y_true==y_pred
            wrong_idx = np.where(wrong==False)[0]
        
            # same and different labels in distance matrix
            # same = 1
            # different = 1e9
            lbls_eq = list()
            lbls_diff = list()
        
            for i in range(y_tr.shape[0]):
                lbls_eq.append(y_tr==y_tr[i])
        
            lbls_eq = np.array(lbls_eq)
            lbls_diff = ~lbls_eq
            
            qsqrt_opposite = np.array(qsqrt)
        
            # diagonal (same element) = 1e9
            for i in range(qsqrt.shape[0]):
                qsqrt[i][i] = 1e9
        
            not_support = list(np.arange(y_tr.shape[0]))
            for i in clf.support_:
                not_support.remove(i)
            not_support = np.array(not_support)
            not_support = not_support.squeeze()
        
            not_wrong = list(np.arange(y_tr.shape[0]))
            for i in wrong_idx:
                not_wrong.remove(i)
            not_wrong = np.array(not_wrong)
            not_wrong = not_wrong.squeeze()
        
            # distance matrix
            # real distances only for the same class
            # 1e9 between SVs
            # 1e9 between opposite classes
            # 1e9 for diagonal (same element)
            dists = np.zeros((qsqrt.shape[0], qsqrt.shape[0]))
            
            for i in range(qsqrt.shape[0]):
                a = (lbls_eq[i] * qsqrt[i]) + (lbls_diff[i] * 1e9)
                dists[i] = a
        
            dists[clf.support_, clf.support_] = 1e9
        
            # 
            for i in not_wrong:
                dists[i,clf.support_] = 1e9
            for i in clf.support_:
                dists[i, wrong_idx] = 1e9
        
            # set wrong_idx with wrong_id and clfs to 1e9 avoid comparison with them
            if(len(not_support) > 0):
                for i in wrong_idx:
                    dists[i,clf.support_] = 1e9
                    for j in wrong_idx:
                        dists[i,j] = 1e9
        
            # opposite matrix
            for i in range(qsqrt_opposite.shape[0]):
                qsqrt_opposite[i] = qsqrt_opposite[i] * lbls_diff[i]
        
            opposite_vectors = list()
            for i in range(qsqrt_opposite.shape[0]):
                opposite_vectors.append(np.flip(qsqrt_opposite[i].argsort(), axis=0))
        
            good_opposites = list(np.arange(0, qsqrt.shape[0]))
       
            # closest wrong elements
            cls_wrong_ord = list()
        
            for i in wrong_idx:
                cls_wrong_ord.append(np.argsort(dists[i,:]))
        
            # cls is the ordered indexes to closest instances for everyone
            cls = list()
            
            for i in range(dists.shape[0]):
                cls.append(np.argsort(dists[i]))
            
            cls = np.array(cls)
            
            ###################### TRAIN ########################
            
            saved = ""
            if(epoch >= 0):
                list_bal_tr.append(bal_acc_tr)
                list_bal_tr_sp.append(bal_acc_tr_reg)

                if(not li_ts is None):
                    list_bal_ts_sp.append(bal_acc_ts_reg)
                    list_bal_ts.append(bal_acc_ts)

                if(not li_val is None):
                    list_bal_val_sp.append(bal_acc_val_reg)
                    list_bal_val.append(bal_acc_val)
        
                #val_loss.append(loss_acc_sv)
                #val_loss_wrong.append(loss_acc)
                #val_loss_opposite.append(loss_opposite)
                nr_sups.append(len(clf.support_))
                
                if(not li_val is None):
                    if(bal_acc_val >= max_val):
                        model_save = {
                                    'model_state': self.model.state_dict(),
                                    'optimizer_state': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler,
                                    'epoch': epoch
                                }
                        
                        #torch.save(model_save, opt.outdir+"/"+opt.outfile+"_model.pth")
                        saved = "[Saved]"
                        max_val = bal_acc_val
                        best_tr = bal_acc_tr
                        best_val = bal_acc_val
                        best_epoch = epoch
                        if(not li_ts is None):
                            best_ts = bal_acc_ts
                else:
                    if(bal_acc_tr >= max_tr):
                        model_save = {
                                    'model_state': self.model.state_dict(),
                                    'optimizer_state': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler,
                                    'epoch': epoch
                                }
                        
                        #torch.save(model_save, opt.outdir+"/"+opt.outfile+"_model.pth")
                        saved = "[Saved]"
                        #max_val = bal_acc_val
                        max_tr = bal_acc_tr
                        #best_ts = bal_acc_ts
                        best_tr = bal_acc_tr
                        #best_val = bal_acc_val
                        best_epoch = epoch


            #pause("Matrices")
        
            self.model.train()
            torch.set_grad_enabled(True)
         
            if(epoch == 0):
                loss_acc_sv = 0.0
                loss_acc = 0.0
                loss_opposite = 0.0
        
            if(epoch == self.epoch-1):
                continue

            ################### BACKPROP SHAKE #####################
       
            if(self.timing):
                start = timer()
        
            if((epoch != 0 and self.shake != 0 and epoch % self.shake == 0) or self.shake == 1):
                loss_opposite = 0.0
        
                if(DEBUG is True):
                    print("Backpropagation Opposite")
        
                for k in range(1):
                    for i in good_opposites:
                        sp_input, sp_label = li_tr.get_tensor(idx=i)
                        X_wrong = self.model(sp_input.to(cuda))
                        reference = torch.tensor(X[opposite_vectors[i][:self.shclose]]).to(cuda)
                        if(self.norm == 1):
                            loss = (1 / ((torch.pow(X_wrong - reference,2).sum(1)).sum()))/(len(good_opposites)*self.shclose)
                        else:
                            loss = (1 / ((torch.pow(X_wrong - reference,2).sum(1)).sum()))
                        loss_tmp = loss.detach().cpu().numpy()
                        loss_opposite += loss_tmp
                        loss.backward()
                        del sp_input, sp_label, X_wrong, reference
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
                    del good_opposites
        
            if((self.shake == 0 or (self.shake != 0 and epoch % self.shake != 0) or epoch == 0) or self.shake == 1):
        
                loss_acc_sv = 0.0
                loss_acc = 0.0
        
                ################### BACKPROP SV #####################
        
                if(DEBUG is True):
                    print("Backpropagation SV")
        
                if(self.svclose > 0):
                    loss = 0
                    loss_old = None
                    self.optimizer.zero_grad()
                    for i in range(clf.support_.shape[0]):
                        sp_input, sp_label = li_tr.get_tensor(idx=clf.support_[i])
                        X_wrong = self.model(sp_input.to(cuda))
                        reference = torch.tensor(X[cls[clf.support_[i]][:self.svclose]]).to(cuda)
                        if(self.norm == 1):
                            loss = (torch.sqrt(torch.pow(X_wrong - reference,2).sum(1)).sum())/(clf.support_.shape[0]*self.svclose)
                        else:
                            loss = (torch.sqrt(torch.pow(X_wrong - reference,2).sum(1)).sum())
                        loss_tmp = loss.detach().cpu().numpy()
                        loss_acc_sv += loss_tmp
                        loss.backward()
                        del sp_input, sp_label, X_wrong, reference
                    self.optimizer.step()
        
                ################### BACKPROP WRONG #####################
                
                if(DEBUG is True):
                    print("Backpropagation Wrong")
        
                if(self.wrclose > 0):
                    self.optimizer.zero_grad()
                    for i in range(wrong_idx.shape[0]):
                        sp_input, sp_label = li_tr.get_tensor(idx=wrong_idx[i])
                        X_wrong = self.model(sp_input.to(cuda))
                        reference = torch.tensor(X[cls_wrong_ord[i][:self.wrclose]]).to(cuda)
                        if(self.norm == 1):
                            loss = (torch.sqrt(torch.pow(X_wrong - reference,2).sum(1)).sum())/(wrong_idx.shape[0]*self.wrclose)
                        else:
                            loss = (torch.sqrt(torch.pow(X_wrong - reference,2).sum(1)).sum())
                        loss_tmp = loss.detach().cpu().numpy()
                        loss_acc += loss_tmp
                        loss.backward()
                        del X_wrong
                        del sp_input, sp_label
                        del reference
                    self.optimizer.step()
                
            ##################################################
       
            if(self.timing):
                end = timer()
                print("Extract   ", extract)
                print("Distance: ", distancetime)
                print("SMO:      ", smotime)
                print("Backprop: ", timedelta(seconds=end-start))
        
            if(epoch >= 0):
                val_loss.append(loss_acc_sv)
                val_loss_wrong.append(loss_acc)
                val_loss_opposite.append(loss_opposite)
        
            print("Loss SV: {:.4f} Loss Wrong: {:.4f} Loss Opposite: {:.4f} Relax: {} {}".format(loss_acc_sv, loss_acc, loss_opposite, self.relax, saved))
            self.scheduler.step()
        
            if(self.optimizer.param_groups[0]['lr'] != current_lr):
                print("New LR: {:05f}".format(self.optimizer.param_groups[0]['lr']))
                current_lr = self.optimizer.param_groups[0]['lr']
        
        del li_tr, li_val, li_ts
        del self.li_tr, self.li_val, self.li_ts

        #return y_pred
        return outputs
