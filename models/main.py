import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(path)
sys.path.append(path)


from fine_tuned_bertweet import BertTweetClassification
from pytorch_pretrained_bert import BertAdam
from config.config import Config
from utils.tweet_preprocessing import tokenizeRawTweetText
from utils.help_functions import batch_iter, pad_sents, sents_to_tensor
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np 
import time


def validation(model, df_val, text_col, label_col, loss_func, val_batch_size = 32):
    """
        :param model: nn.Module, the model being trained
        :param df_val: validation set
        :param loss_func: nn.Module, loss function 
        @returns avg loss value across validation dataset
    """
    model.eval()
    tweets = list(df_val[text_col])
    labels = list(df_val[label_col])

    num_val_samles = df_val.shape[0]
    n_batch = int(np.ceil(num_val_samles/val_batch_size))

    total_loss = 0

    with torch.no_grad():
        for i in range(n_batch):
            sents = tweets[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(labels[i*val_batch_size:(i+1)*val_batch_size], dtype= torch.long).to(Config.device)

            batch_size = len(sents)
            output = model(sents)[0]
            batch_loss = loss_func(output, targets)
            total_loss = batch_loss.item() * batch_size
    
    return total_loss/num_val_samles

def evaluate()



def fit(model, df_train, df_val,  text_col, label_col, n_epochs = 10, lr=1e-3, max_grad_norm=1.0, train_batch_size=64, val_batch_size=128,
    bert_config='', display_num = 20, valid_niter=10, max_epoch = 5):
  

    train_label = dict(df_train[label_col].value_counts())
    label_max = float(max(train_label.values()))
    print("train label: ", train_label)
    train_label_weight = torch.tensor([label_max/train_label[i] for i in train_label])
    
    # print("Type: ", train_label_weight.dtype)
    # print(train_label_weight)

    optimizer_grouped_parameters = [
        {'params': model.model.bert.parameters()},
        {'params': model.model.classifier.parameters(), 'lr': lr}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr = lr, max_grad_norm = max_grad_norm)
    criterion = torch.nn.CrossEntropyLoss(weight = train_label_weight.float().to(Config.device), reduction = 'mean')
    model.train()

    train_iter = patience = cum_loss = report_loss = 0
    total_samples = display_samples = epoch = 0
    valid_loss_hist = []
    train_time = begin_time = time.time()
    print('Begin training...')

    while True:
        epoch += 1
        print(".....Epoch {}".format(epoch))
        for batch in batch_iter(df_train, batch_size = train_batch_size, shuffle = False):
            train_iter +=1 
            sents = list(batch[text_col])
            labels= np.array(batch[label_col])
            batch_size = len(sents)
            labels = torch.tensor(labels, dtype = torch.long).to(Config.device)
            optimizer.zero_grad()
            
            output = model(sents)
            loss = criterion(output[0], labels) #calculate loss
            
            loss.backward() #back prop
            optimizer.step() #update weights           
            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            display_samples += batch_size
            total_samples += batch_size

        # if train_iter % display_num == 0:
        print('epoch %d, iter %d, avg. loss %.2f, '
                'total samples %d, speed %.2f samples/sec, '
                'time elapsed %.2f sec' % 
                (epoch, train_iter, report_loss / display_samples,
                total_samples, display_samples / (time.time() - train_time),
                time.time() - begin_time))
        train_time = time.time()
        report_loss = display_samples = 0.

        #perform validation 
        # print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % 
        #               (epoch, train_iter, cum_loss / total_samples, total_samples))
        cum_loss = total_samples = 0.0
        print('begin validation ...')

        valid_loss = validation(model, df_val, text_col, label_col, criterion, val_batch_size=val_batch_size)                
        print('validation: iter %d, loss %f' % (train_iter, valid_loss))
        improved_loss = len(valid_loss_hist)==0 or valid_loss < min(valid_loss_hist)
        valid_loss_hist.append(valid_loss)
        if improved_loss:
            patience = 0
#             print('save currently the best model to [%s]' % args['--model']+'_model.bin', file=sys.stderr)
#             model.save(args['--model']+'_model.bin')

#             # also save the optimizers' state
#             torch.save(optimizer.state_dict(), args['--model'] + '.optim')
        else: #if valid loss did not improve
            patience += 1
            # print('hit patience %d out of %d' % (patience, int(args['--patience'])), file=sys.stderr)
            print('early termination!')
            exit(0)
#             if patience >= int(args['--patience']):
#                 num_restarts += 1
#                 print('hit #%d restart out of max %d restarts' % (num_restarts, int(args['--max-num-trial'])), file=sys.stderr)
#                 if num_restarts >= int(args['--max-num-trial']):
#                     print('early termination!', file=sys.stderr)
#                     exit(0)

#                 # decay lr, and restore from previously best checkpoint
#                 lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
#                 print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

#                 # load model
#                 params = torch.load(args['--model'], map_location=lambda storage, loc: storage)
#                 model.load_state_dict(params['state_dict'])

#                 print('restore parameters of the optimizers', file=sys.stderr)
#                 optimizer.load_state_dict(torch.load(args['--model'] + '.optim'))
#
#                 # set new lr
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr
#
#                 # reset patience
#                 patience = 0

        if epoch == max_epoch:
            print('reached maximum number of epochs!')
            exit(0)


if __name__ == "__main__":
    # read input file:
    input_path = Config.input_path
    data = pd.read_csv(input_path)
    data = data[Config.selected_columns]

    
    # preprocessing text
    data[Config.prepro_text] = data[Config.text].apply(lambda x: tokenizeRawTweetText(x))
    data[Config.prepro_text] = data[Config.prepro_text].apply(lambda x: '<s> '+x)
    data[Config.explan] = data[Config.explan].replace(np.nan, '', regex = True)
    data[Config.prepro_explan] = data[Config.explan].apply(lambda x: tokenizeRawTweetText(x))
    

    # assign labels to numbers
    labels = set(data[Config.label])
    data[Config.prepro_label] = data[Config.label].astype('category').cat.codes


    data['len'] = data[Config.prepro_text].apply(lambda x: len(x.split(" ")))
    #remove very short tweets len<=3
    data = data[data['len']>3] 
    
    #split train/valid/test
    train_data, test_data = train_test_split(data, test_size = Config.test_size, stratify = data[Config.prepro_label], random_state=Config.random_seed)
    train_data, valid_data = train_test_split(train_data, test_size = Config.test_size, stratify = train_data[Config.prepro_label], random_state=Config.random_seed)

    model = BertTweetClassification(num_class = len(set(data[Config.prepro_label])), bert_config='vinai/bertweet-base', device=Config.device)

    fit(model, train_data, valid_data, Config.prepro_text, Config.prepro_label)