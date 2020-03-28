import numpy as np
# from run import predict
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def predict(model, user_id, train, item_ids=None):
    """
    Make predictions for evaluation: given a user id, it will
    first retrieve the test sequence associated with that user
    and compute the recommendation scores for items.

    Parameters
    ----------

    user_id: int
        users id for which prediction scores needed.
    item_ids: array, optional
        Array containing the item ids for which prediction scores
        are desired. If not supplied, predictions for all items
        will be computed.
    """

    # if self.test_sequence is None:
    #     raise ValueError('Missing test sequences, cannot make predictions')

    # set model to evaluation model
    model.eval()
    with torch.no_grad():
        test_sequences_np = train.test_sequences.sequences[user_id, :]
        # sequences_np = self.test_sequence.sequences
        test_sequences_np = np.atleast_2d(test_sequences_np)

        if item_ids is None:
            item_ids = np.arange(train.num_items).reshape(-1, 1)

        sequences = torch.from_numpy(test_sequences_np).long()
        item_ids = torch.from_numpy(item_ids).long()
        user_id = torch.from_numpy(np.array([[user_id]])).long()

        user, sequences, items = (user_id.to(device),
                                  sequences.to(device),
                                  item_ids.to(device))

        out = model(sequences, user, items, for_pred=True)

    return out.cpu().numpy().flatten()


def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train_matrix = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()

    # evaluation 满就是因为这里， 是一个一个用户预测的

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue
        # argsort 返回从小到大的下标
        predictions = -predict(model, user_id, train)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train_matrix[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(
                targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, mean_aps
if __name__ == "__main__":
    actual = [[1, 2], [3, 4, 5]]
    predicted = [[10, 20, 1, 30, 40], [3, 10, 20, 4, 5]]

    apks=[]
    for i in range(len(actual)):

        apks.append(_compute_apk(actual[i], predicted[i],2))
    mean_aps = np.mean(apks)
    print(mean_aps)
