import torch
import time
from typing import Literal

#the only modes that can be called are train and val
_MODES = Literal['train', 'val']

def GetMean(x,y):
    return x / y

class Logger(object):
    def __init__(self, mode, batches):
        self.mode = mode
        self.batches = batches

    def __call__(self,batch,loss,timer,accuracy):
        batch = batch + 1
        batchString = "{} {}/{} |".format(self.mode.upper(),batch,self.batches)
        lossString = " Loss: {:.4f} |".format(loss)
        timeString = " Batch Time: {:.3f} Seconds |".format(timer.getDuration())
        accuracyString = " Accuracy: {:.4f}".format(accuracy)
        outputString = batchString + lossString + timeString + accuracyString
        print(outputString)
        
class BatchTimer:
    def __init__(self):
        self.start_time = None
        self.duration = None
    
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.duration = time.time() - self.start_time
        
    def getDuration(self):
        return self.duration

        
def acc(x,y):
    _, pred = torch.max(x,1)
    return (pred == y).float().mean()


def train(model, lossFunc, dLoader, mode: _MODES, optimizer = None, scheduler = None, device = 'cpu'):
    training = True if mode == 'train' else False
    log = Logger(mode,len(dLoader))
    loss = 0
    epTime = 0
    epAcc = 0
    
    timer = BatchTimer()
    for batch, (x,y) in enumerate(dLoader):
        timer.start()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        batchLoss = lossFunc(pred,y)
        
        # if the model is train
        if training:
            batchLoss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        timer.stop()
        
        batchLoss = batchLoss.detach().cpu()
        loss += batchLoss
    
        log(batch, batchLoss, timer,acc(pred,y))
        epTime += timer.getDuration()
        epAcc += acc(pred,y)
    
    if training and scheduler is not None:
        scheduler.step()
    
    loss = loss / (1 + batch)
    
    print("\nAverage | Loss: {:.4f} | Time: {:.3f} | Accuracy: {:.4f} ".format(loss,epTime,
                                                                                     GetMean(epAcc,len(dLoader))))
       
