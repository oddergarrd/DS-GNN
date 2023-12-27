import pickle
from torch import optim
import argparse
import os
import pandas as pd
import numpy as np
import warnings
import tqdm
from tqdm import trange

from model_piece import *

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--dropout', type=float, default='0.2',
                    help='dropout rate')
parser.add_argument('--rnn_T', type=int, default='15',
                    help='rnn_length')
parser.add_argument('--batch_size', type=int, default='1',
                    help='batch_size')
parser.add_argument('--clip', type=float, default='0.0025',
                    help='rnn clip')
parser.add_argument('--relation', type=str, default='attention_gate'
                    )
parser.add_argument('--save', type=bool, default=True,
                    help='save model')


def load_dataset(DEVICE, label):
    with open('./data/stock_feature_Normalized_21_1041.pkl', 'rb') as handle:
        stock_feature = pickle.load(handle)
        stock_feature.to(torch.float)
    with open('./data/y_close_' + str(label) + '.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/chain_code_dict_1041_final.pkl', 'rb') as handle:
        chain_info = pickle.load(handle)
    with open('./data/adj_all_1_1191.pkl', 'rb') as handle:
        relation_static_1 = pickle.load(handle)
    with open('./data/adj_all_2_1191.pkl', 'rb') as handle:
        relation_static_2 = pickle.load(handle)
    with open('./data/adj_all_3_1191.pkl', 'rb') as handle:
        relation_static_3 = pickle.load(handle)

    print('DEVICE:', DEVICE)
    stock_feature = stock_feature
    y_load = y_load
    relation_static_1 = relation_static_1.to(DEVICE)
    relation_static_2 = relation_static_2.to(DEVICE)
    relation_static_3 = relation_static_3.to(DEVICE)

    relation_dict = {
        '1': relation_static_1,
        '2': relation_static_2,
        '3': relation_static_3,
    }
    cluster_num = len(chain_info.keys())
    return stock_feature, y_load, relation_dict, chain_info, cluster_num


def train(model, x_train, train_y, tensor_adjacency, rnn_length_x, cluster_info, cluster_num,
          batch_size, alpha):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length_x:]
    total_loss = 0
    total_loss_count = 0
    batch_train = batch_size  # batch_size的设置

    for i in train_seq:
        x_train_i = x_train[:][i - rnn_length_x:i].to(device)
        with torch.cuda.amp.autocast():
            output, loss_global = model(tensor_adjacency, x_train_i, cluster_info, cluster_num)
            loss_predict = criterion(output, train_y[i - 1].long().to(device))
            loss = alpha * loss_predict + loss_global
            # loss = loss_predict
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    torch.cuda.empty_cache()

    return total_loss / total_loss_count


def evaluate(model, x_eval, y_eval, tensor_adjacency, rnn_length_x, cluster_info, cluster_num):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length_x:]
    preds = []
    trues = []
    preds_possible = -torch.ones(len(seq), x_eval.shape[1], 2)
    preds_save = -torch.ones(len(seq), x_eval.shape[1])
    trues_save = -torch.ones(len(seq), x_eval.shape[1])
    total_loss = 0
    total_loss_count = 0
    for i in seq:
        x_eval_i = x_eval[:][i - rnn_length_x:i].to(device)

        with torch.no_grad():
            output, loss_global = model(tensor_adjacency, x_eval_i, cluster_info, cluster_num)
        loss = criterion(output, y_eval[i - 1].long().to(device)).to(device)
        total_loss += loss.item()
        total_loss_count += 1
        output = output.detach().cpu()
        preds_save[i - rnn_length_x] = output.argmax(-1)
        preds_possible[i - rnn_length_x] = output
        trues_save[i - rnn_length_x] = y_eval[i - 1].long()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i - 1].cpu().numpy())

    acc, auc, mcc, f1, kappa = metrics1(trues, preds)
    return {"acc": acc, "auc": auc, "mcc": mcc, "f1": f1, "kappa": kappa,
            "loss": total_loss / total_loss_count}, preds_save, trues_save, preds_possible


def save_preds(best_preds, best_trues, best_preds_possible, file_name='eval'):
    cl_file = "./model_result/all_ours_direction_pred/" + file_name + ".pkl"  # 保存预测结果和概率
    output = open(cl_file, 'wb')
    pickle.dump((best_preds, best_trues, best_preds_possible), output)
    output.close()


def save_record(dicts, seed, file_name='train'):
    dicts.to_excel('./model_result/all_ours_direction_loss/' + file_name + '__' + str(
        seed) + '.xlsx')  # 保存train_loss_history,eval_acc_history,eval_auc_history。


if __name__ == '__main__':
    args = parser.parse_args(args=[])
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)
    print(device)
    lr_list = [0.0001]
    dim_list = [64]
    d_piece_list = [8]
    alpha_list = [2]
    labels = ['0.01']
    for label in labels:
        for d_piece in d_piece_list:
            for alpha in alpha_list:
                for dim in dim_list:
                    for lrate in lr_list:
                        for i in range(0, 1):  # 一次循环训练一个模型
                            seed = random.randint(1, 100000)
                            set_seed(seed)
                            stock_feature, y, relation_dict, cluster_info, cluster_num = load_dataset(device, label)
                            print('stock_feature.shape:', stock_feature.shape)
                            print('y.shape:', y.shape)
                            print('relation_dict.shape:', relation_dict['1'].shape)
                            print('cluster_num:', cluster_num)
                            idx = np.random.permutation(stock_feature.shape[1])
                            shuf_feats = stock_feature[:, idx, :]
                            global num_stock
                            num_stock = stock_feature.shape[1]

                            relation = args.relation
                            rnn_hidden_size = dim
                            GNN_hidden_size = dim

                            model = GcnNet(d_markets=stock_feature.shape[-1],
                                           rnn_hidden_size=rnn_hidden_size,
                                           GNN_hidden_size=GNN_hidden_size,
                                           d_piece=d_piece,
                                           num_stock=num_stock,
                                           device=device,
                                           num_layers=2,
                                           )
                            model.to(device)

                            print("model:", model.state_dict().keys())
                            optimizer = optim.Adam(model.parameters(), lr=lrate)
                            criterion = torch.nn.NLLLoss()
                            rnn_length_x = args.rnn_T  # 以前t天的基本面信息得到当天的股票节点特征
                            batch_size = args.batch_size
                            days = stock_feature.shape[0]

                            x_train = stock_feature[:int(days / 5 * 4)]
                            x_eval = stock_feature[
                                     int(days / 5 * 4) - rnn_length_x: int(days / 10 * 9)]  # int(days/4*3)
                            x_test = stock_feature[int(days / 10 * 9) - rnn_length_x:]

                            y_train = y[:int(days / 5 * 4)]
                            y_eval = y[int(days / 5 * 4) - rnn_length_x:int(days / 10 * 9)]
                            y_test = y[int(days / 10 * 9) - rnn_length_x:]

                            MAX_EPOCH = 300
                            best_model_file = 0
                            epoch = 0
                            wait_epoch = 0
                            wait_epoch_acc = 0
                            eval_epoch_best = 0
                            eval_epoch_acc = 0
                            best_train_loss = 10000

                            eval_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            train_loss_history = pd.DataFrame()
                            train_loss_history1 = []
                            for epoch in trange(MAX_EPOCH):
                                train_loss = train(model,
                                                   x_train,
                                                   y_train,
                                                   relation_dict,
                                                   rnn_length_x,
                                                   cluster_info,
                                                   cluster_num,
                                                   batch_size,
                                                   alpha
                                                   )

                                train_loss_history1.append(train_loss)
                                eval_dict, preds_save, trues_save, preds_possible = evaluate(model,
                                                                                             x_eval,
                                                                                             y_eval,
                                                                                             relation_dict,
                                                                                             rnn_length_x,
                                                                                             cluster_info,
                                                                                             cluster_num,
                                                                                             )
                                eval_df = eval_df.append(eval_dict, ignore_index=True)
                                test_dict, preds_save_test, trues_save_test, preds_possible_test = evaluate(model,
                                                                                                            x_test,
                                                                                                            y_test,
                                                                                                            relation_dict,
                                                                                                            rnn_length_x,
                                                                                                            cluster_info,
                                                                                                            cluster_num,
                                                                                                            )
                                test_df = test_df.append(test_dict, ignore_index=True)
                                eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_f1{:.4f}," \
                                           "test_auc{:.4f},test_acc{:.4f},test_f1{:.4f}".format(
                                    epoch, train_loss, eval_dict['auc'], eval_dict['acc'], eval_dict['f1'],
                                    test_dict['auc'], test_dict['acc'], test_dict['f1'])
                                print(eval_str)

                                if (eval_dict['f1'] > eval_epoch_best):
                                    eval_epoch_best = eval_dict['f1']
                                    best_train_loss = train_loss
                                    eval_best_str = "relation{},epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, " \
                                                    "eval_f1{:.4f},test_auc{:.4f},test_acc{:.4f},test_f1{:.4f}".format(
                                        relation, epoch, train_loss, eval_dict['auc'], eval_dict['acc'],
                                        eval_dict['f1'], test_dict['auc'],
                                        test_dict['acc'], test_dict['f1'])
                                    wait_epoch = 0
                                    if args.save:
                                        if best_model_file:
                                            os.remove(best_model_file)
                                            os.remove(best_record_file1)
                                            os.remove(best_record_file2)
                                        best_model_file = "./SavedModels/ours/relu_label{}_batch{}_alpha{}_RNN_T_{}_dim{}_d_piece{}_seed{}_lr{" \
                                                          "}_eval,auc{:.4f}_acc{:.4f}_f1{:.4f}_mcc{:.4f}_kappa{:.4f}_test,auc{:.4f}_acc{:.4f}_f1{:.4f}_mcc{:.4f}_kappa{:.4f}_epoch{" \
                                                          "}.pkl".format(
                                            label, batch_size, alpha, args.rnn_T, dim, d_piece, seed, lrate,
                                            eval_dict['auc'], eval_dict['acc'], eval_dict['f1'], eval_dict['mcc'],
                                            eval_dict['kappa'],
                                            test_dict['auc'], test_dict['acc'], test_dict['f1'], test_dict['mcc'],
                                            test_dict['kappa'],
                                            epoch, )
                                        torch.save(model.state_dict(), best_model_file)

                                        eval_name = relation + "_" + str(lrate) + "_" + str(
                                            dim) + ",seed{},eval,auc{:.4f}_f1{:.4f}_mcc{:.4f}_epoch{}_".format(
                                            seed, eval_dict['auc'], eval_dict['f1'], eval_dict['mcc'], epoch)
                                        test_name = relation + "_" + str(lrate) + "_" + str(
                                            dim) + ",seed{},test,auc{:.4f}_f1{:.4f}_mcc{:.4f}_epoch{}_".format(
                                            seed, test_dict['auc'], test_dict['f1'], test_dict['mcc'], epoch)
                                        best_record_file1 = "./model_result/all_ours_direction_pred/" + eval_name + ".pkl"
                                        best_record_file2 = "./model_result/all_ours_direction_pred/" + test_name + ".pkl"
                                        save_preds(preds_save, trues_save, preds_possible, eval_name)
                                        save_preds(preds_save_test, trues_save_test, preds_possible_test, test_name)

                                elif train_loss < best_train_loss:
                                    wait_epoch += 0
                                else:
                                    wait_epoch += 1

                                if wait_epoch > 100:
                                    print("saved_model_result:", eval_best_str)
                                    break

                                if (eval_dict['acc'] != eval_epoch_acc):
                                    eval_epoch_acc = eval_dict['acc']
                                    wait_epoch_acc = 0
                                else:
                                    wait_epoch_acc += 1

                                if wait_epoch_acc > 15:
                                    print("saved_model_result:", eval_best_str)
                                    break
                                epoch += 1

                            train_loss_history["train_loss_history"] = train_loss_history1
                            save_record(train_loss_history, seed, 'train')
