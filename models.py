import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from scipy import sparse
from layers import *
from metrics import *


flags = tf.app.flags
FLAGS = flags.FLAGS


class IMCN(object):
    def __init__(self, placeholders, n_inst, input_dim, transfer_list, adj_list, node_wgt_list, train_idx, mat01_tr_te):
        self.placeholders = placeholders
        self.n_inst = n_inst
        self.n_class = placeholders['labels'].get_shape().as_list()[1]
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.transfer_list = transfer_list
        self.adj_list = adj_list
        self.node_wgt_list = node_wgt_list
        self.train_idx = train_idx
        self.mat01_tr_te = mat01_tr_te

        self.W_node_wgt = tf.Variable(tf.random_uniform([FLAGS.max_node_wgt, FLAGS.node_wgt_embed_dim],
                                                        minval=-math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim)),
                                                        maxval=math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim))),
                                      name="W_node_wgt")

        self.HGCNlayers = []
        self.GCNlayers = []     # store H = GH+G
        self.HGCNactivations = []
        self.out_hgcn = []
        self.GCN2layers = []
        self.GCN2activations = []
        self.out_gcn2 = []

        self.hgcn_centroid = tf.zeros(shape=(self.n_class, self.output_dim))  # clss * out_dim
        self.gcn2_centroid = tf.zeros(shape=(self.n_class, self.output_dim))  # clss * out_dim
        self.opt_op = None

        self.outs = []
        self.loss = 0.
        self.l_ce = 0.
        self.l_cn = 0.
        self.l_cnu = 0.
        self.l_cns = 0.
        self.l_se = 0.      # s: mlp   t: hgcn
        self.l_ds = 0.      # difference between node-to-class distributions of s and t

        self.build()

    def _build(self):
        # HGCN structure
        FCN_hidden_list = [FLAGS.hidden] * 100
        node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[0])
        self.HGCNlayers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FCN_hidden_list[0],
                                                placeholders=self.placeholders,
                                                support=self.adj_list[0] * FLAGS.channel_num,
                                                transfer=self.transfer_list[0],
                                                node_emb=node_emb,
                                                mod='input',
                                                layer_index=0,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=True))  # G0

        for i in range(FLAGS.coarsen_level - 1):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[i + 1])
            self.HGCNlayers.append(GraphConvolution(input_dim=FCN_hidden_list[i],
                                                    output_dim=FCN_hidden_list[i + 1],
                                                    placeholders=self.placeholders,
                                                    support=self.adj_list[i + 1] * FLAGS.channel_num,
                                                    transfer=self.transfer_list[i + 1],
                                                    node_emb=node_emb,
                                                    mod='coarsen',
                                                    layer_index=i + 1,
                                                    act=tf.nn.relu,
                                                    dropout=True))  # Gi

        for i in range(FLAGS.coarsen_level, FLAGS.coarsen_level * 2):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[2*FLAGS.coarsen_level - i])
            self.HGCNlayers.append(GraphConvolution(input_dim=FCN_hidden_list[i - 1],
                                                    output_dim=FCN_hidden_list[i],
                                                    placeholders=self.placeholders,
                                                    support=self.adj_list[2*FLAGS.coarsen_level - i] * FLAGS.channel_num,
                                                    transfer=self.transfer_list[2*FLAGS.coarsen_level - 1 - i],
                                                    node_emb=node_emb,
                                                    mod='refine',
                                                    layer_index=i,
                                                    act=tf.nn.relu,
                                                    dropout=True))  # G?-1

        self.HGCNlayers.append(GraphConvolution(input_dim=FCN_hidden_list[FLAGS.coarsen_level * 2 - 1],
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                support=self.adj_list[0] * FLAGS.channel_num,
                                                transfer=self.transfer_list[0],
                                                node_emb=0,
                                                mod='output',
                                                layer_index=FLAGS.coarsen_level * 2,
                                                act=lambda x: x,
                                                dropout=True))

        # outputs of HGCN
        self.HGCNactivations.append(self.inputs)
        for i in range(len(self.HGCNlayers)):
            layer = self.HGCNlayers[i]
            hidden, pre_GCN = layer(self.HGCNactivations[-1])
            self.GCNlayers.append(pre_GCN)
            if i >= FLAGS.coarsen_level and i < FLAGS.coarsen_level * 2:
                hidden = hidden + self.GCNlayers[FLAGS.coarsen_level * 2 - i - 1]
            self.HGCNactivations.append(hidden)
        self.out_hgcn = tf.nn.l2_normalize(self.HGCNactivations[-1], dim=1)

        # 2-layer MLP
        self.GCN2activations.append(self.inputs)
        self.GCN2layers.append(GraphConvolution2(input_dim=self.input_dim,
                                                 output_dim=FCN_hidden_list[0],
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.relu,
                                                 dropout=True,
                                                 sparse_inputs=True))
        layer = self.GCN2layers[-1]
        hidden = layer(self.inputs)
        self.GCN2activations.append(hidden)

        self.GCN2layers.append(GraphConvolution2(input_dim=FCN_hidden_list[0],
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: x,
                                                 dropout=True))
        layer = self.GCN2layers[-1]
        hidden = layer(self.GCN2activations[-1])
        self.GCN2activations.append(hidden)
        self.out_gcn2 = tf.nn.l2_normalize(self.GCN2activations[-1], dim=1)

        self.outs = tf.nn.l2_normalize((1-FLAGS.h_mlp_percent) * self.out_hgcn + FLAGS.h_mlp_percent * self.out_gcn2, dim=1)

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.HGCNlayers)):
            for var in self.HGCNlayers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for i in range(len(self.GCN2layers)):
            for var in self.GCN2layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # l_ce: cross-entropy classification loss
        self.l_ce = masked_softmax_cross_entropy(self.outs, self.placeholders['labels'],
                                                 self.placeholders['labels_mask'])

        # l_cnu and l_cns: contrastive loss
        self.contrastiveloss_modify()

        # l_se: semantic centroid alignment of labeled nodes
        # l_ds: distribution difference of unlabeled nodes
        self.semanticloss()

        # Final loss
        self.l_cn = 1 * self.l_cnu + 1 * self.l_cns
        self.loss += self.l_ce + FLAGS.lam1 * self.l_cn + FLAGS.lam2 * self.l_se + FLAGS.lam3 * self.l_ds

    def semanticloss(self):
        fea_s = self.out_gcn2
        fea_t = self.out_hgcn

        # hidden embeddings and labels of training set
        y_train = tf.gather(self.placeholders['labels'], self.train_idx, axis=0)
        fea_s_train = tf.nn.l2_normalize(tf.gather(fea_s, self.train_idx, axis=0), dim=1)
        fea_t_train = tf.nn.l2_normalize(tf.gather(fea_t, self.train_idx, axis=0), dim=1)

        # number of samples in each class for training set
        n_sample_classes = tf.reduce_sum(y_train, axis=0)
        ones_num = tf.ones_like(n_sample_classes)
        n_sample_classes = tf.where(n_sample_classes < 1.0, ones_num, n_sample_classes)

        # calculating centroids, sum and divide
        d = fea_s.shape[1]
        s_sum_feature = tf.matmul(tf.transpose(y_train), fea_s_train)
        n_sample_classes_ma = tf.matmul(tf.reshape(n_sample_classes, [-1, 1]), tf.ones([1, d], dtype=tf.float32))
        current_s_centroid = tf.div(s_sum_feature, n_sample_classes_ma)

        t_sum_feature = tf.matmul(tf.transpose(y_train), fea_t_train)
        n_sample_classes_ma = tf.matmul(tf.reshape(n_sample_classes, [-1, 1]), tf.ones([1, d], dtype=tf.float32))
        current_t_centroid = tf.div(t_sum_feature, n_sample_classes_ma)

        # Moving Centroid
        self.gcn2_centroid = (1 - FLAGS.class_decay) * self.gcn2_centroid + FLAGS.class_decay * current_s_centroid
        self.hgcn_centroid = (1 - FLAGS.class_decay) * self.hgcn_centroid + FLAGS.class_decay * current_t_centroid

        # loss: centroid alignment with labeled training set
        self.l_se = tf.reduce_sum(tf.square(self.gcn2_centroid - self.hgcn_centroid))  # reduce_mean

        # loss: node-class distribution with unlabeled nodes
        idx = np.arange(int(fea_s.shape[0]))
        unlabeled_idx = np.delete(idx, self.train_idx)
        fea_s_unlabeled = tf.gather(fea_s, unlabeled_idx, axis=0)
        fea_t_unlabeled = tf.gather(fea_t, unlabeled_idx, axis=0)
        s_c = tf.nn.softmax(tf.matmul(fea_s_unlabeled, tf.transpose(self.gcn2_centroid)) / FLAGS.temperature)
        t_c = tf.nn.softmax(tf.matmul(fea_t_unlabeled, tf.transpose(self.hgcn_centroid)) / FLAGS.temperature)
        zeros_tf = tf.zeros_like(s_c)
        s_c = tf.where(s_c < 1e-4, zeros_tf, s_c)
        t_c = tf.where(t_c < 1e-4, zeros_tf, t_c)
        self.l_ds = tf.reduce_mean(tf.log(tf.pow(s_c, -t_c)))       # reduce_mean

    def contrastiveloss(self):
        # compute the unsupervised contrastive loss
        # 公式4
        cos_dist = tf.exp(tf.matmul(self.out_gcn2, tf.transpose(self.out_hgcn)) / 0.5)
        neg = tf.reduce_mean(cos_dist, axis=1)
        diag_cos = tf.diag_part(cos_dist)
        positive_sum = diag_cos
        pos_neg1 = positive_sum / neg
        # 公式5
        cos_dist = tf.exp(tf.matmul(self.out_hgcn, tf.transpose(self.out_gcn2)) / 0.5)
        neg = tf.reduce_mean(cos_dist, axis=1)
        diag_cos = tf.diag_part(cos_dist)
        positive_sum = diag_cos
        pos_neg2 = positive_sum / neg
        # 公式3
        pos_neg3 = tf.concat([pos_neg1, pos_neg2], 0)
        self.l_cnu = - tf.reduce_mean(tf.log(pos_neg3))

        # compute the supervised contrastive loss
        # 公式7
        h1 = tf.gather(self.out_gcn2, self.train_idx, axis=0)
        h2 = tf.gather(self.out_hgcn, self.train_idx, axis=0)
        h_cos = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
        supervised_positive_sum = tf.reduce_sum(h_cos * self.mat01_tr_te[0], axis=1)
        supervised_negative_sum = (tf.reduce_sum(h_cos * self.mat01_tr_te[1], axis=1)
                                   + supervised_positive_sum) / (np.shape(self.train_idx)[0] - 1)
        supervised_positive_sum /= np.sum(self.mat01_tr_te[0], axis=1)
        pos_neg_sup_1 = supervised_positive_sum / supervised_negative_sum
        # 公式8
        h2 = tf.gather(self.out_gcn2, self.train_idx, axis=0)
        h1 = tf.gather(self.out_hgcn, self.train_idx, axis=0)
        h_cos = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
        supervised_positive_sum = tf.reduce_sum(h_cos * self.mat01_tr_te[0], axis=1)
        supervised_negative_sum = (tf.reduce_sum(h_cos * self.mat01_tr_te[1], axis=1)
                                   + supervised_positive_sum) / (np.shape(self.train_idx)[0] - 1)
        supervised_positive_sum /= np.sum(self.mat01_tr_te[0], axis=1)
        pos_neg_sup_2 = supervised_positive_sum / supervised_negative_sum
        # 公式6
        pos_neg_sup_3 = tf.concat([pos_neg_sup_1, pos_neg_sup_2], 0)
        self.l_cns = -tf.reduce_mean(tf.log(pos_neg_sup_3))

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outs, self.placeholders['labels'], self.placeholders['labels_mask'])

    def build(self):
        self._build()
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softmax(self.outs)

    def contrastiveloss_modify(self):
        # compute the unsupervised contrastive loss
        u_cos_dist = tf.exp(tf.matmul(self.out_gcn2, tf.transpose(self.out_hgcn)) / 0.5)
        u_pos = tf.reduce_sum(tf.diag_part(u_cos_dist))
        u_neg = tf.reduce_sum(u_cos_dist) #- u_pos
        self.l_cnu = - tf.log(u_pos/u_neg)#/int(u_cos_dist.shape[0])

        # compute the supervised contrastive loss
        h1 = tf.gather(self.out_gcn2, self.train_idx, axis=0)
        h2 = tf.gather(self.out_hgcn, self.train_idx, axis=0)
        s_cos_dist = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
        s_pos = tf.reduce_sum(s_cos_dist * self.mat01_tr_te[0])
        s_neg = tf.reduce_sum(s_cos_dist)# - s_pos
        self.l_cns = -tf.log(s_pos/s_neg)#/int(s_cos_dist.shape[0])  # tf.cast(tf.reduce_sum(self.mat01_tr_te[0]), tf.float32)

    # def contrastiveloss_modify(self):
    #     # compute the unsupervised contrastive loss
    #     u_sim_11 = tf.exp(tf.matmul(self.out_gcn2, tf.transpose(self.out_gcn2)) / 0.5)
    #     u_sim_22 = tf.exp(tf.matmul(self.out_hgcn, tf.transpose(self.out_hgcn)) / 0.5)
    #     u_sim_12 = tf.exp(tf.matmul(self.out_gcn2, tf.transpose(self.out_hgcn)) / 0.5)
    #
    #     u_pos_11 = tf.reduce_sum(tf.diag_part(u_sim_11))
    #     u_pos_22 = tf.reduce_sum(tf.diag_part(u_sim_22))
    #     u_pos_12 = tf.reduce_sum(tf.diag_part(u_sim_12))
    #
    #
    #     u_pos = tf.reduce_sum(tf.diag_part(u_cos_dist))
    #     u_neg = tf.reduce_sum(u_cos_dist) #- u_pos
    #     self.l_cnu = - tf.log(u_pos/u_neg)/int(u_cos_dist.shape[0])
    #
    #     # compute the supervised contrastive loss
    #     h1 = tf.gather(self.out_gcn2, self.train_idx, axis=0)
    #     h2 = tf.gather(self.out_hgcn, self.train_idx, axis=0)
    #     s_cos_dist = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
    #     s_pos = tf.reduce_sum(s_cos_dist * self.mat01_tr_te[0])
    #     s_neg = tf.reduce_sum(s_cos_dist)# - s_pos
    #     self.l_cns = -tf.log(s_pos/s_neg)/int(s_cos_dist.shape[0])