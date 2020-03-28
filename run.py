import argparse
from time import time

import torch.optim as optim
from torch.autograd import Variable

from interactions import Interactions
from utils import *
from collections import defaultdict
from eval_metrics import *

from perec import perec
import heapq

import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def evaluation(model, train, test, config,topk=20):

    test_dict = defaultdict(lambda: [])
    for i, user_id in enumerate(test.user_ids):

        test_dict[user_id].append(test.item_ids[i])
    test_set = list(test_dict.values())
    topk = max(list(map(len, test_set)))+100
    print("test_set max length for user", topk-100)

    num_users = train.num_users
    num_items = train.num_items
    batch_size = 10
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    train_matrix = train.tocsr()
    test_sequences = train.test_sequences.sequences


    # +++++++++++++++++++++++++++++++++++++++++
    if config.baseline=="pop":
        predict = np.sum(train_matrix, axis=0)
        predict = np.array(predict).reshape(-1)
        print("predict",predict.shape)
        # item_score_dict = {}
        # for j in range(num_items):
        #     # item = test_item[i][j]
        #     item_score_dict[j] = predict[item]
        
        pred_list=[]
        pred_list_per_user=list(predict)
        for i in range(num_users):
            pred_list.append(pred_list_per_user)
        
        pred_list = np.array(pred_list)
        pred_list[train_matrix[user_indexes].toarray() > 0] = -np.inf
        
        print("pred_list",pred_list.shape)
        pred_list = np.argsort(pred_list)
        for i in range(len(pred_list)):
            pred_list[i]=pred_list[i][::-1]
        
        # print(type(pred_list))


        # print("pred_list",pred_list[:4])
        
        precision, recall, ndcg = [], [],  []
        for k in [1, 5, 10]:
            precision.append(precision_at_k(test_set, pred_list, k))
            recall.append(recall_at_k(test_set, pred_list, k))

            ndcg.append(ndcg_k(test_set, pred_list, k))

        return precision, recall, ndcg



    # ++++++++++++++++++++++++++++++++++
    gate_score_list=[]
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)

        batch_test_sequences = torch.from_numpy(
            batch_test_sequences).type(torch.LongTensor).to(device)
        item_ids = torch.from_numpy(item_indexes).type(
            torch.LongTensor).to(device)
        batch_user_ids = torch.from_numpy(
            np.array(batch_user_index)).type(torch.LongTensor).to(device)

        # 这里计算target item 和他们之间的
        rating_pred = model(batch_test_sequences,
                            batch_user_ids, item_ids, True)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = -np.inf

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(
            arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[
            :, None], arr_ind_argsort]

        # rating_pred= -rating_pred
        # print("rating_pred",rating_pred.shape)
        # batch_pred_list=np.argsort(rating_pred, axis=1)[:,:20]
        # print("batch_pred_list", batch_pred_list.shape, batch_pred_list[:3])

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
        # gate_score_list.append(gate_score)
        
    # print("gate_score_list", gate_score_list)
    # gate_score_list=np.array(gate_score_list)
    # gate_score_list.tofile("gate_socre")
    print("pred_list", pred_list[1710])

    precision, recall, ndcg = [], [], []
    for k in [1, 5, 10]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))

        ndcg.append(ndcg_k(test_set, pred_list, k))
    return precision, recall, ndcg


def generate_negative_samples(users, train, neg_num, candidate={}):
    """
    Sample negative from a candidate set of each user. The
    candidate set of each user is defined by:
    {All Items} \ {Items Rated by User}

    Parameters
    ----------

    users: array of np.int64
        sequence users
    interactions: :class:`spotlight.interactions.Interactions`
        training instances, used for generate candidates
    n: int
        total number of negatives to sample for each sequence
    """

    users_ = users.squeeze()
    negative_samples = np.zeros((users_.shape[0], neg_num), np.int64)
    if not candidate:
        all_items = np.arange(train.num_items - 1) + 1  # 0 for padding
        train = train.tocsr()
        for user, row in enumerate(train):
            candidate[user] = list(set(all_items) - set(row.indices))

    for i, u in enumerate(users_):
        for j in range(neg_num):
            x = candidate[u]
            negative_samples[i, j] = x[
                np.random.randint(len(x))]

    return negative_samples


def train_model(train, test, config):
    """
    The general training loop to fit the model

    Parameters
    ----------

    train: :class:`spotlight.interactions.Interactions`
        training instances, also contains test sequences
    test: :class:`spotlight.interactions.Interactions`
        only contains targets for test sequences
    verbose: bool, optional
        print the logs
    """

    # convert to sequences, targets and users
    sequences_np = train.sequences.sequences
    targets_np = train.sequences.targets
    users_np = train.sequences.user_ids.reshape(-1, 1)

    L, T = train.sequences.L, train.sequences.T

    n_train = sequences_np.shape[0]

    output_str = 'total training instances: %d' % n_train
    print(output_str)

    num_items = train.num_items
    num_users = train.num_users

    # test_sequence = train.test_sequences
    if config.baseline=="pop":
        t_pop=time()
        precision, recall, MAP, ndcg = evaluation(
                None, train, test,config)
        logger.info(', '.join(str(e) for e in precision))
        logger.info(', '.join(str(e) for e in recall))
        logger.info(', '.join(str(e) for e in MAP))
        logger.info(', '.join(str(e) for e in ndcg))
        logger.info("Evaluation time:{}".format(time() - t_pop))
        return 

    if config.baseline == "perec":
        model = perec(num_users, num_items, config).to(device)

    optimizer = optim.Adam(
        model.parameters(), weight_decay=config.l2, lr=config.learning_rate)

    candidate = {}

    start_epoch = 0

    for epoch_num in range(start_epoch, config.n_iter):

        t1 = time()

        # set model to training mode
        model.train()

        users_np, sequences_np, targets_np = shuffle(
            users_np, sequences_np, targets_np)

        negatives_np = generate_negative_samples(
            users_np, train, config.neg_samples, candidate)

        # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
        users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                torch.from_numpy(
                                                    sequences_np).long(),
                                                torch.from_numpy(
                                                    targets_np).long(),
                                                torch.from_numpy(negatives_np).long())

        users, sequences, targets, negatives = (users.to(device),
                                                sequences.to(device),
                                                targets.to(device),
                                                negatives.to(device))

        epoch_loss = 0.0

        for (minibatch_num,
                (batch_users,
                 batch_sequences,
                 batch_targets,
                 batch_negatives)) in enumerate(minibatch(users,
                                                          sequences,
                                                          targets,
                                                          negatives,
                                                          batch_size=config.batch_size)):
            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            items_prediction = model(batch_sequences,
                                     batch_users,
                                     items_to_predict)

            (targets_prediction,
                negatives_prediction) = torch.split(items_prediction,
                                                    [batch_targets.size(1),
                                                     batch_negatives.size(1)], dim=1)

            optimizer.zero_grad()
            if config.bpr == 1:
                # compute the BPR loss  这里需要修改
                loss = -torch.log(torch.sigmoid(targets_prediction -
                                                negatives_prediction) + 1e-8)
                loss = torch.mean(torch.sum(loss))

            else:
                # compute the binary cross-entropy loss
                positive_loss = -torch.mean(
                    torch.log(torch.sigmoid(targets_prediction)))
                negative_loss = -torch.mean(
                    torch.log(1 - torch.sigmoid(negatives_prediction)))
                loss = positive_loss + negative_loss
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= minibatch_num + 1

        t2 = time()
        output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (
            epoch_num + 1, t2 - t1, epoch_loss, time() - t2)
        print(output_str)
        t3 = time()

        if (epoch_num + 1) % config.epoch_eval == 0:
            model.eval()
            precision, recall, ndcg = evaluation(
                model, train, test, config)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    # parser.add_argument('--train_root', type=str, default='datasets/ml1m/test/train.txt')
    # parser.add_argument('--test_root', type=str, default='datasets/ml1m/test/test.txt')
    parser.add_argument('--dataset', type=str, default='ml1m')

    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--epoch_eval', type=int, default=1)
    parser.add_argument('--baseline', type=str, default="lspr")

    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')
    parser.add_argument('--bpr', type=int, default='1')

    config = parser.parse_args()

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # load dataset
    train_root = "datasets/{}/test/train.txt".format(config.dataset)
    train = Interactions(train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)
    print("train max item id", max(train.item_ids), "min",
          min(train.item_ids), "num", train.num_items)
    print("train item map", len(train.item_map))

    test_root = 'datasets/{}/test/test.txt'.format(config.dataset)
    test = Interactions(test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print("test max item id", max(test.item_ids), "min",
          min(test.item_ids), "num", test.num_items)
    # print("3243",test.item_map['3243'])
    print("train item map", len(test.item_map))

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_model(train, test, config)
