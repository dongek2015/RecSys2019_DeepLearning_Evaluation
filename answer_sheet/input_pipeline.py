import numpy as np

def generate_negatives(neg_users, true_mat, item_range, sort=False, use_trick=False):
    """
    Generate negative samples for data augmentation
    """
    neg_u = []
    neg_i = []

    # If using the shortcut, generate negative items without checking if the associated
    # user has interacted with it. Speeds up training significantly with very low impact
    # on accuracy.
    if use_trick:
        neg_items = cp.random.randint(0, high=item_range, size=neg_users.shape[0])
        return neg_users, neg_items

    # Otherwise, generate negative items, check if associated user has interacted with it,
    # then generate a new one if true
    while len(neg_users) > 0:
        neg_items = cp.random.randint(0, high=item_range, size=neg_users.shape[0])
        neg_mask = true_mat[neg_users, neg_items]
        neg_u.append(neg_users[neg_mask])
        neg_i.append(neg_items[neg_mask])

        neg_users = neg_users[cp.logical_not(neg_mask)]

    neg_users = cp.concatenate(neg_u)
    neg_items = cp.concatenate(neg_i)

    if not sort:
        return neg_users, neg_items

    sorted_users = cp.sort(neg_users)
    sort_indices = cp.argsort(neg_users)

    return sorted_users, neg_items[sort_indices]

class DataGenerator():
    """
    Class to handle data augmentation
    """
    def __init__(self,
                 seed,
                 num_users,                 # type: int
                 num_items,                 # type: int
                 train_users,               # type: np.ndarray
                 train_items,               # type: np.ndarray
                 train_labels,              # type: np.ndarray
                 train_batch_size,          # type: int
                 eval_users,            # type: np.ndarray
                 eval_items,            # type: np.ndarray
                 eval_labels,
                 eval_users_per_batch,      # type: int
                ):
        np.random.seed(seed)
        self.num_users = num_users
        self.num_items = num_items
        self._train_users = np.copy(train_users)
        self._train_items = np.copy(train_items)
        # we use -1 or 0 as 1, 1 as 0, because 1 is 80%
        self._train_labels = train_labels
        self.train_batch_size = train_batch_size
        self._eval_users = eval_users
        self._eval_items = eval_items
        self._eval_labels = eval_labels
        self.eval_users_per_batch = eval_users_per_batch

        # Eval data
        self.eval_users = None
        self.eval_items = None
        self.dup_mask = None

        # Training data
        self.train_users_batches = None
        self.train_items_batches = None
        self.train_labels_batches = None

    # Augment test data with negative samples
    def prepare_eval_data(self):
        pos_eval_users = cp.array(self._pos_eval_users)
        pos_eval_items = cp.array(self._pos_eval_items)

        neg_mat = cp.array(self._neg_mat)

        neg_eval_users_base = cp.repeat(pos_eval_users, self._eval_negative_samples)

        # Generate negative samples
        test_u_neg, test_i_neg = generate_negatives(neg_users=neg_eval_users_base, true_mat=neg_mat,
                                                    item_range=self.num_items, sort=True, use_trick=False)

        test_u_neg = test_u_neg.reshape((-1, self._eval_negative_samples)).get()
        test_i_neg = test_i_neg.reshape((-1, self._eval_negative_samples)).get()

        test_users = self._pos_eval_users.reshape((-1, 1))
        test_items = self._pos_eval_items.reshape((-1, 1))
        # Combine positive and negative samples
        test_users = np.concatenate((test_u_neg, test_users), axis=1)
        test_items = np.concatenate((test_i_neg, test_items), axis=1)

        # Generate duplicate mask
        ## Stable sort indices by incrementing all values with fractional position
        indices = np.arange(test_users.shape[1]).reshape((1, -1)).repeat(test_users.shape[0], axis=0)
        summed_items = np.add(test_items, indices/test_users.shape[1])
        sorted_indices = np.argsort(summed_items, axis=1)
        sorted_order = np.argsort(sorted_indices, axis=1)
        sorted_items = np.sort(test_items, axis=1)
        ## Generate duplicate mask
        dup_mask = np.equal(sorted_items[:,0:-1], sorted_items[:,1:])
        dup_mask = np.concatenate((dup_mask, np.zeros((test_users.shape[0], 1))), axis=1)
        r_indices = np.arange(test_users.shape[0]).reshape((-1, 1)).repeat(test_users.shape[1], axis=1)
        dup_mask = dup_mask[r_indices, sorted_order].astype(np.float32)

        # Reshape all to (-1) and split into chunks
        batch_size = self.eval_users_per_batch * test_users.shape[1]
        split_indices = np.arange(batch_size, test_users.shape[0]*test_users.shape[1], batch_size)
        self.eval_users = np.split(test_users.reshape(-1), split_indices)
        self.eval_items = np.split(test_items.reshape(-1), split_indices)
        self.dup_mask = np.split(dup_mask.reshape(-1), split_indices)

        # Free GPU memory to make space for Tensorflow
        cp.get_default_memory_pool().free_all_blocks()

    # Augment training data with negative samples
    def prepare_train_data(self):
        batch_size = self.train_batch_size

        shuffled_order = np.random.permutation(self._train_users.shape[0])
        self._train_users = self._train_users[shuffled_order]
        self._train_items = self._train_items[shuffled_order]
        self._train_labels = self._train_labels[shuffled_order]

        # Manually create batches
        split_indices = np.arange(batch_size, self._train_users.shape[0], batch_size)
        self.train_users_batches = np.split(self._train_users, split_indices)
        self.train_items_batches = np.split(self._train_items, split_indices)
        self.train_labels_batches = np.split(self._train_labels, split_indices)