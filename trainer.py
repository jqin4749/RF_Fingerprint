import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import models as md
import data_loader as dl 
import os, pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class manger:
    def __init__(self, mode = 'ova') -> None:
        self.dev_ =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.max_runs_ = 10
        self.max_epochs_ = 10
        self.batch_size_ = 64
        self.threshold_ = 0.1
        self.delta_ = 0.05
        self.mode_ = mode
        self.base_results_dir_ = '/home/jack/Desktop/rf_fingerprint/results/%s_test'%mode
        self.dataset_ = dl.dataset()

        return
    
    def run_given_AK(self, A_size, K_size):
        run_dir = os.path.join(self.base_results_dir_, 'A%dK%d'%(A_size, K_size))
        os.makedirs(run_dir, exist_ok=True)
        cur_dir = os.getcwd()
        os.chdir(run_dir)
        self.dataset_.update_AK(A_size, K_size)
        self.run()
        os.chdir(cur_dir)
    
    def run(self):
        for run in range(self.max_runs_):
            self.dataset_.shuffle()
            self.fit(run)

    def one_hot(self, y):
        num_classes = self.dataset_.A_size_
        label_to_return = np.zeros((len(y),num_classes))
        for idx, i in enumerate(y):
            if i < num_classes:
                label_to_return[idx,i] = 1
        return torch.from_numpy(label_to_return)

    def fit(self, id):
        data = self.dataset_.get_train_test_set()
        model = md.ova(self.dataset_.A_size_) if self.mode_ == 'ova' else md.autoencoder()
        model.train()
        opt = torch.optim.Adam(model.parameters(),lr=1e-3)
        model.to(self.dev_)
        freq_weights = torch.Tensor(self.get_label_frequency_weight(data['train_y'])).to(self.dev_)
        loss_list = []
        for ep in range(self.max_epochs_):
            train_idx = np.random.choice(len(data['train_x']), self.batch_size_, replace = False)
            x = torch.FloatTensor(data['train_x'][train_idx]).to(self.dev_)
            y = self.one_hot(data['train_y'][train_idx].reshape(-1)).float().to(self.dev_)
            if self.mode_ == 'ova':
                loss_to_accu = 0
                for idx, l in enumerate(y.T):
                    if torch.sum(l) == 0:
                        continue
                    opt.zero_grad()
                    out = model(idx, x)
                    if torch.sum(l) != 0:
                        loss = F.binary_cross_entropy(out.view(-1,1), l.view(-1,1),weight=freq_weights[idx])
                    else:
                        loss = F.binary_cross_entropy(out.view(-1,1), l.view(-1,1), weight=freq_weights[-1]) # for the outlier
                    loss.backward()
                    opt.step()
                    loss_to_accu += loss.item()
                loss_list.append(loss_to_accu)
            else:
                opt.zero_grad()
                out = model(x)
                loss = F.mse_loss(out, x)
                loss.backward()
                opt.step()
                loss_list.append(loss.item())

            print('Training epoch %d, loss: %f'%(ep, loss.item()))
        if self.mode_ != 'ova':
            self.threshold_ = np.quantile(loss_list[-10:],0.9)
            self.delta_ = np.std(loss_list[-10:])
            print('[Autoencoder] Set the threshold to : %f, delta to: %f'%(self.threshold_,self.delta_))


        self.test(model, data, 'same_day_run%d'%id)
        self.test(model, self.dataset_.get_diff_day_test_set(), 'different_day_run%d'%id)
        plt.figure()
        plt.plot(loss_list)
        plt.title('Training loss of run %d'%id)
        plt.savefig('loss_run%d'%id,dpi=400)  

    def get_label_frequency_weight(self, y):
        total = self.dataset_.A_size_ + 1
        count = np.zeros(total)
        for i in y:
            count[i] += 1
        weight = (count - np.min(count)) / (np.max(count) - np.min(count)) * 0.8 + 0.1 # 0.1 ~ 0.9
        return list(1.0 - weight)
    
    def test(self, model, data, name):
        model.eval()
        label = list(data['test_y'].reshape(-1))
        pred = []
        loss_list = []
        for x in np.array_split(data['test_x'],int(np.ceil(len(data['test_x'])/self.batch_size_)),axis=0):
            x = torch.FloatTensor(x).to(self.dev_)
    
            if self.mode_ == 'ova':
                out_ = [model(idx, x) for idx in range(self.dataset_.A_size_)]
                out_ = np.concatenate([o.detach().cpu().numpy() for o in out_],axis=-1)
                for out in out_:
                    out = [(idx,o) for idx, o in enumerate(out) if o > self.threshold_]
                    if len(out) == 0:
                        pred.append(self.dataset_.A_size_)
                    else:
                        getd = lambda x: x[1]
                        out.sort(key=getd)
                        pred.append(out[-1][0])
            else:
                out_ = model(x)
                for out, org in zip(out_, x):
                    loss = F.mse_loss(out, org)
                    loss_list.append(loss.item())
                    if loss.item() > self.threshold_ :
                        pred.append(1) # anomaly detected
                    else:
                        pred.append(0) # normal sample

        acc_count = []
        anomaly_label = np.zeros(len(label))
        anomaly_label[[idx for idx, i in enumerate(label) if i == self.dataset_.A_size_]] = 1
        if self.mode_ == 'ova':
            acc_count = [p for p, l in zip(pred,label) if p == l]   
        else:
            acc_count = [p for p, l in zip(pred,anomaly_label) if p == l]

        acc = len(acc_count) / len(label)     
        print('Test on %s, accuracy: %f'%(name, acc))
        anomaly_label = list(anomaly_label)
        prob_false_alarm, prob_detection = self.cal_prob_metric(pred, anomaly_label)
        print('Prob. False Alarm: %f, Prob. Detection: %f'%(prob_false_alarm, prob_detection))
        res = {
            'A_size': self.dataset_.A_size_,
            'acc': acc,
            'pred': pred,
            'label': label,
            'anomaly_label': anomaly_label,
            'prob_false_alarm': prob_false_alarm,
            'prob_detection': prob_detection
        }
        with open('./res_%s.pkl'%name, 'wb') as f:
            pickle.dump(res, f)   
        if self.mode_ != 'ova':
            plt.figure()
            plt.plot(loss_list)
            plt.axhline(y = self.threshold_, color = 'r', linestyle = 'dashed')
            plt.axhline(y = self.threshold_+self.delta_, color = 'r', linestyle = 'dashed')
            plt.axhline(y = self.threshold_-self.delta_, color = 'r', linestyle = 'dashed')
            plt.title('Test loss of run (autoencoder)')
            plt.savefig('test_loss_%s'%name,dpi=400)
        return 

    def cal_prob_metric(self, pred, label):
        if len([i for i in pred if i != 0 and i != 1]) != 0:
            pred_ = np.zeros(len(pred))
            pred_[[idx for idx, i in enumerate(pred) if i == self.dataset_.A_size_]] = 1
            pred = list(pred_)
        
        confu_matrix = confusion_matrix(label, pred, normalize='all')
        # print('Confusion Matrix:')
        # print(confu_matrix)
        return confu_matrix[0,1], confu_matrix[1,1]