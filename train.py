from __future__ import division
from __future__ import print_function
import warnings
import time
import copy
import scipy.io as sio
from utils import *
from models import HGCN
from coarsen import *
warnings.filterwarnings('ignore')


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# dataset
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # cora, citeseer, pubmed, photo, cs, computer, physics
flags.DEFINE_integer('public', 0, '1: 20 pre class; 0: for label rate')
flags.DEFINE_float('percent', 0.005, 'Label rate.')

flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')        # 0.01
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')  # 0.9~0.1
flags.DEFINE_float('weight_decay', 0.05, 'Weight for L2 loss on embedding matrix.')     # 0.05

# parameters for computing loss
flags.DEFINE_float('class_decay', 0.3, 'decay of updating class centroids.')
flags.DEFINE_float('temperature', 0.1, 'temperature for similarity between node embedding and class centroids.')
flags.DEFINE_float('lam1', 0, 'Weight for node-wise consistency.')
flags.DEFINE_float('lam2', 1, 'Weight for class centroid difference.')
flags.DEFINE_float('lam3', 1.5, 'Weight for distribution difference.')
flags.DEFINE_float('h_mlp_percent', 0.3, 'h_mlp_percent in the final embedding.')

# parameters for coarsening the graph
flags.DEFINE_integer('node_wgt_embed_dim', 5, 'Number of units for node weight embedding.')
flags.DEFINE_integer('coarsen_level', 4, 'Maximum coarsen level.')
flags.DEFINE_integer('max_node_wgt', 282, 'Maximum node_wgt to avoid super-node being too large.')  # 50,120,282
flags.DEFINE_integer('channel_num', 4, 'Number of channels')
flags.DEFINE_integer('seed', 123, 'Number of epochs to train.')


# Define model evaluation function
def evaluate(features, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Graph Coarsening.
graph, mapping = read_graph_from_adj(FLAGS.dataset)
original_graph = graph
transfer_list = []
adj_list = [copy.copy(graph.A)]
node_wgt_list = [copy.copy(graph.node_wgt)]

for i in range(FLAGS.coarsen_level):
    match, coarse_graph_size = generate_hybrid_matching(FLAGS.max_node_wgt, graph)
    coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
    transfer_list.append(copy.copy(graph.C))
    graph = coarse_graph
    adj_list.append(copy.copy(graph.A))
    node_wgt_list.append(copy.copy(graph.node_wgt))
    # print('There are %d nodes in the %d coarsened graph' % (graph.node_num, i+1))

for i in range(len(adj_list)):
    adj_list[i] = [preprocess_adj(adj_list[i])]

best = 0.0
best_test_acc_mat = []
for iter in range(1, 11):
    np.random.seed(FLAGS.seed + iter)
    tf.set_random_seed(FLAGS.seed + iter)

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_rate(FLAGS.dataset, FLAGS.public,
                                                                                            FLAGS.percent, FLAGS.seed+iter)

    n_inst = np.shape(y_train)[0]
    n_class = np.shape(y_train)[1]

    # Some preprocessing
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    num_supports = 1

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # label vector: labels of training set are real classes, others are -1
    y_real_train = np.argmax(y_train, axis=1)
    y_real = np.ones([n_inst]) * -1
    train_idx = np.argwhere(np.sum(y_train, axis=1) > 0)[:, 0]  # train_idx vector
    y_real[train_idx] = y_real_train[train_idx]

    # sample indexes in each class
    intra_class_idx = []
    for i in range(n_class):
        intra_class_idx.append(np.argwhere(y_real == i)[:, 0])

    # 2708*2708 mat: whether two samples in training set have same label
    train_mat01 = CalCLass01Mat(y_train, train_mask)

    # two 140*140 matrices: mark two are or not from same class
    mats_intra_inter = CalIntraClassMat01(y_real[train_idx])
    num_labeled = int(np.sum(y_train))
    mats_intra_inter[0] += np.eye(num_labeled)

    # Create model
    model = HGCN(placeholders, n_inst=n_inst, input_dim=features[2][1], transfer_list=transfer_list, adj_list=adj_list,
                 node_wgt_list=node_wgt_list, train_idx=train_idx, mat01_tr_te=mats_intra_inter)

    # Initialize session
    sess = tf.Session()
    # Init variables
    np.random.seed(FLAGS.seed + iter)
    tf.set_random_seed(FLAGS.seed + iter)
    sess.run(tf.global_variables_initializer())

    # train model
    cost_train = []
    acc_train = []
    cost_val = []
    acc_val = []
    cost_test = []
    acc_test = []

    best_epoch = 0
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, FLAGS.epochs+1):
        t = time.time()

        # Training
        feed_dict = construct_feed_dict(features, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.l_ce, model.l_se, model.l_ds, model.l_cn,
                         model.out_hgcn, model.out_gcn2, model.outs],
                        feed_dict=feed_dict)
        cost_train.append(outs[1])
        acc_train.append(outs[2])

        # Validation
        cost, val_acc, duration = evaluate(features, y_val, val_mask, placeholders)
        cost_val.append(cost)
        acc_val.append(val_acc)

        # Test
        test_cost, test_acc, test_duration = evaluate(features, y_test, test_mask, placeholders)
        cost_test.append(test_cost)
        acc_test.append(test_acc)

        if test_acc > best_test_acc:
            best_epoch = epoch
            best_test_acc = test_acc

            # if best_test_acc > best:
            #     best = best_test_acc
            #     sio.savemat("h_hgcn.mat", {"array": np.array(outs[7])})
            #     sio.savemat("h_gcn2.mat", {"array": np.array(outs[8])})
            #     sio.savemat("h_final.mat", {"array": np.array(outs[9])})

        # Print results
        print("Iter:", '%02d' % iter,
              "Epoch:", '%04d' % epoch,
              "train_loss=", "{:.5f}".format(outs[1]),
              "l_ce=", "{:.5f}".format(outs[3]),
              "l_se=", "{:.5f}".format(outs[4]),
              "l_ds=", "{:.5f}".format(outs[5]),
              "l_cn=", "{:.5f}".format(outs[6]),
              "train_acc=", "{:.5f}".format(outs[2]),
              "val_acc=", "{:.5f}".format(val_acc),
              "test_acc=", "{:.5f}".format(test_acc),
              "best_acc=", "{:.5f}".format(best_test_acc), )

    print('** Best test accuracy ({:02d}):  {:.5f}\n'.format(best_epoch, best_test_acc))
    best_test_acc_mat.append(best_test_acc)

print("\n------------------------------------------------")
print("Final test acc:")
print("mean acc: {:.5f}, std: {:.5f}.".format(np.mean(best_test_acc_mat), np.std(best_test_acc_mat)))
print("best one: {:.5f}.".format(np.max(best_test_acc_mat)))
print("------------------------------------------------")


