#se va afisa:
#(terminal separat)cat se foloseste din cpu
#(terminal curent):
#dizpozitivul pe care se face training-ul si disp pe care se face testul
#tipul retelei folosita
#
#se face leanring-ul
#
#automat isi depoziteaza weight-urile dupa ce le a invatat
#se afiseaza(train loss,acc|valid loss,acc)(average pe toate epocile) test loss,acc 
#timpul in care a facut learining-ul
#se plolteaza train_acc si test_acc imrpeuna cu train_loss si test_loss
#se poate rula cu optiunea already_learnedpentru a icnarca paramterii in retea si a rula direct evacc pe test
#alta optiune la parameterul din train alreadyParameters va antrena autmoat modelul si va stoca parametrii invatati intr-un fisier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)








from colorama import Fore, Back, Style
import subprocess
import time
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
batch_size, lrate, num_epochs = 256, 0.5, 50
network_typeName="ResNet"

class ResBlock(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels,conv1_1=False):
        super().__init__()
        self.conv1x1=None
        self.conv1 = nn.Conv2d(input_channels, input_channels,
                               kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels)

        if conv1_1:
            self.conv1x1 = nn.Conv2d(input_channels, input_channels,
                               kernel_size=1,stride=1, padding=0)

    def forward(self, X):
        y_std_beforeRelu=nn.ReLU()(self.bn1(self.conv1(X)))
        y_std_afterRelu=self.bn1(self.conv1(y_std_beforeRelu))
        if self.conv1x1:
            return nn.ReLU()(y_std_afterRelu+self.conv1x1(X))
        else:
            return nn.ReLU()(y_std_afterRelu+X)

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

def load_data_svhn(batch_size, resize=None):
    """Download the SVHN dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # Load the SVHN dataset with 'split' instead of 'train'
    svhn_train = torchvision.datasets.SVHN(
        root="../data", split='train', transform=trans, download=True)
    svhn_test = torchvision.datasets.SVHN(
        root="../data", split='test', transform=trans, download=True)
    
    # Optionally, you can split the training set into training and validation sets
    svhn_train, svhn_val = torch.utils.data.random_split(svhn_train, [43257, 30000],
                                                         generator=torch.Generator().manual_seed(42))
    
    return (torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(svhn_val, batch_size=batch_size, shuffle=False, num_workers=2),
            torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=2))

def plot_loss(train_loss_all, val_loss_all):
    epochs = range(1, len(train_loss_all) + 1)
    plt.plot(epochs, train_loss_all, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_all, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_acc_all, val_acc_all):
    epochs = range(1, len(train_acc_all) + 1)
    plt.plot(epochs, train_acc_all, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_all, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)
            total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100

def train_epoch(net, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += float(l)
        total_hits += sum(y_hat.argmax(axis=1).type(y.dtype) == y)
        total_samples += y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100

def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device,device2,alreadyParameters):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    loss = nn.CrossEntropyLoss()
   
    if(alreadyParameters=='already_learned'):
        net.to(device2)
        net.load_state_dict(torch.load('ResNet_{}parameters.pth'.format(num_epochs)))
        test_loss, test_acc = evaluate_accuracy(net, test_iter, loss, device2)
        print("\n\nParameters already computed MODE:\n\n")
        print("TestDevice: set to    {}".format(device2.type))
        print("Network type: "+Fore.MAGENTA+network_typeName+Style.RESET_ALL+"\n\n\n\n\n")
        print("Test({} datasets): ({})loss | ({})%, acc".format(len(test_iter.dataset),test_loss,test_acc))
        return None,None,None,None
    

    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.to(device)
    print("\n\nParameters in computing MODE:\n\n")
    print("TrainingDevice: set to    {}".format(device.type))
    print("TestDevice: set to    {}\n".format(device2.type))
    print("Network type: "+Fore.MAGENTA+network_typeName+Style.RESET_ALL)
    print(torch.cuda.get_device_name(device) if device.type == "cuda" else
          torch.cuda.get_device_name(device2) if device2.type == "cuda" else "")
    print("\n\n\n\n")

    #learning phase
    startExecTime=time.time()
    for epoch in range(num_epochs):
        print("Learning phase: "+Fore.LIGHTGREEN_EX,epoch,Style.RESET_ALL+" out of "+Fore.GREEN,num_epochs,Style.RESET_ALL,end='\r')
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        #print(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}') 
    endExecTime=time.time()
    torch.save(net.state_dict(), 'ResNet_{}parameters.pth'.format(num_epochs))#save the parameters

    test_loss, test_acc = evaluate_accuracy(net, test_iter, loss, device2)
    avg_train_loss = float(sum(train_loss_all) /epoch) 
    avg_train_acc = float(sum(train_acc_all) /epoch) 
    avg_val_loss = float(sum(val_loss_all) /epoch) 
    avg_val_acc = float(sum(val_acc_all) /epoch) 
    print("Training({} datasets): ({})loss | ({})%, acc".format(len(train_iter.dataset),avg_train_loss,avg_train_acc))
    print("Valid({} datasets): ({})loss | ({})%, acc".format(len(val_iter.dataset),avg_val_loss,avg_val_acc))
    print("Test({} datasets): ({})loss | ({})%, acc".format(len(test_iter.dataset),test_loss,test_acc))
    print(Fore.RED+Back.GREEN+"Total time spent on learning:{}minutes".format(float((endExecTime-startExecTime)/60)))
    print(Style.RESET_ALL)

    plot_loss(train_loss_all,val_loss_all)
    plot_accuracy(train_acc_all,val_acc_all)
    return train_loss_all, train_acc_all, val_loss_all, val_acc_all









if __name__ == '__main__':
    #====================================================================================
    #definim datasetul ,il imaprtim pe training valid si test cu batch-ul de 256
    #subprocess.run(['gnome-terminal', '/home/eduard/Facultate/PI/sapt4', '--', 'bash', '-c', 'auto-cpufreq --stats; exec bash'])
    train_iter, val_iter, test_iter = load_data_svhn(batch_size)
    #definim reteaua si in interiorul fucntiei de antreanre setam si loss fct si optimizer si initializam w cu xavier
    #vom obtine  train si valid, loss fucntion si accuracy pe fiecare epoca pt trainingIter si validIter
    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2), 
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=3, stride=2),#6x6
        ResBlock(64),ResBlock(64),nn.Conv2d(64,128,kernel_size=1),
        ResBlock(128,True),ResBlock(128),nn.Conv2d(128,256,kernel_size=1),
        ResBlock(256,True),ResBlock(256),nn.Conv2d(256,512,kernel_size=1),
        ResBlock(512,True),ResBlock(512),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),nn.Linear(512*1*1,10)
    )
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, test_iter, num_epochs, lrate, try_gpu(0),try_gpu(0),'already_learned') 

