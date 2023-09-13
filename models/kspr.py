import multiprocessing
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from time import time 

import numpy as np
from sklearn.linear_model import LinearRegression, enet_path, Lasso


def kspr_parallel_func(X, y, y_permute, num_class, permute_strategy,
                       reduce_alg, threshold, spr_mode):
    kspr = KSPR(num_class=num_class, reduce_alg=reduce_alg, reduce_dim=num_class, 
                permute_strategy=permute_strategy, threshold=threshold, spr_mode=spr_mode)
    clean_set, q = kspr.predict(X, y, y_permute=y_permute)
    return clean_set, q

def kspr_parallel(X, y, y_permute=None, num_class=None, clean_set=None, 
                  num_classes_sub=None, permute_strategy=None, num_examples_sub=None,
                  reduce_alg=None, threshold=None, spr_mode=None):
    idx = np.arange(len(X))
    res_list = []
    super2sub = []
    num_classes_sub = num_classes_sub

    if num_classes_sub == num_class:
        super2sub = [np.arange(num_class).tolist()]
    else:
        if clean_set is None:
            proto = []
            for i in range(num_class):
                proto.append(np.mean(X[y==i], axis=0, keepdims=True))
            proto = np.concatenate(proto)
        else:
            proto = np.zeros((num_class, X.shape[1]))
            count = np.zeros(num_class)
            for i in range(len(X)):
                if int(idx[i]) in clean_set:
                    count[y[i]] += 1
                    proto[y[i]] += X[i]
            proto = proto / count.reshape(-1,1)
        similarity = np.dot(proto, proto.T) 
        candidate = np.arange(num_class).tolist()
    
        while len(candidate) > 0:
            sub = []
            x = candidate[0]
            sub.append(x)
            candidate.remove(x)
            for i in range(num_classes_sub-1):
                # select the new class by the smallest sum of similarities.
                sim2sub = similarity[sub][:, candidate].sum(0)
                if permute_strategy == 'random':
                    x = candidate[sim2sub.argmin()]
                else:
                    x = candidate[sim2sub.argmax()]
                sub.append(x)
                candidate.remove(x)
            sub.sort()
            super2sub.append(sub)
        
    num_per_task = num_examples_sub

    X_list = []
    Y_list = []
    Y_permute_list = []
    indexes2full_list = []
    class_num_list = []
    for sub in super2sub:
        sub_stats = defaultdict(dict)
        min_num = 100000000000
        max_num = 0
        for l in sub:
            selected_indexes = y==l
            sub_stats[l]['idx'] = idx[selected_indexes]
            sub_stats[l]['X'] = X[selected_indexes]
            sub_stats[l]['y'] = y[selected_indexes]
            sub_stats[l]['y_permute'] = y_permute[selected_indexes]
            min_num = min(len(sub_stats[l]['idx']), min_num)
            max_num = max(len(sub_stats[l]['idx']), max_num)
        total_num = max_num
        for i in range(0, total_num, num_per_task):   
            indexes2full = {}
            X_sub = []
            Y_sub = []
            Y_permute_sub = []
            for ind, l in enumerate(sub):
                num_sub = len(sub_stats[l]['X'])
                if i+num_per_task < num_sub or i >= num_sub: 
                    X_sub.append(sub_stats[l]['X'][i%num_sub:(i+num_per_task)%num_sub])
                    Y_sub.append(sub_stats[l]['y'][i%num_sub:(i+num_per_task)%num_sub])
                    Y_permute_sub.append(sub_stats[l]['y_permute'][i%num_sub:(i+num_per_task)%num_sub])
                else:
                    X_sub.append(sub_stats[l]['X'][i:])
                    X_sub.append(sub_stats[l]['X'][:(i+num_per_task)%num_sub])
                    Y_sub.append(sub_stats[l]['y'][i:])
                    Y_sub.append(sub_stats[l]['y'][:(i+num_per_task)%num_sub])
                    Y_permute_sub.append(sub_stats[l]['y_permute'][i:])
                    Y_permute_sub.append(sub_stats[l]['y_permute'][:(i+num_per_task)%num_sub])
                for j in range(num_per_task):
                    indexes2full[num_per_task*ind+j]=sub_stats[l]['idx'][(i+j)%num_sub]
            X_sub = np.concatenate(X_sub)
            Y_sub = np.concatenate(Y_sub)
            Y_permute_sub = np.concatenate(Y_permute_sub)[:, sub]
            for ind, s in enumerate(sub):
                Y_sub[Y_sub==s] = ind
            Y_permute_sub_new = []
            Y_permute_sub_argsort = Y_permute_sub.argsort(axis=-1)
            for ind, label in enumerate(Y_sub):
                if Y_permute_sub_argsort[ind,-1] == label:
                    Y_permute_sub_new.append(Y_permute_sub_argsort[ind,-2])
                else:
                    Y_permute_sub_new.append(Y_permute_sub_argsort[ind,-1])
            Y_permute_sub = np.array(Y_permute_sub_new)

            X_list.append(X_sub)
            Y_list.append(Y_sub)
            Y_permute_list.append(Y_permute_sub)
            indexes2full_list.append(indexes2full)
            class_num_list.append(len(sub))
    pool = Pool(processes=int(multiprocessing.cpu_count()))
    for i in range(len(X_list)):
        res = pool.apply_async(func=kspr_parallel_func, args=(
            X_list[i], Y_list[i], Y_permute_list[i], class_num_list[i], permute_strategy, reduce_alg, threshold, spr_mode))
        res_list.append(res)
    pool.close()
    pool.join()
    clean_set = []
    q_list = []
    for i, res in enumerate(res_list):
        sub_set, q = res.get()
        q_list += q
        for j in sub_set:
            clean_set.append(indexes2full_list[i][j])
    clean_set = set(clean_set)
    return clean_set, q_list

class KSPR(object):

    def __init__(self, 
                 classifier='lr', 
                 reduce_alg='none',
                 reduce_dim=None,
                 num_class=None, 
                 permute_strategy='random',
                 threshold=None,
                 spr_mode=None,
                 ):
        self.num_class = num_class
        self.permute_strategy = permute_strategy
        self.threshold = threshold if threshold is not None else 1.0
        self.spr_mode = spr_mode
        self.lr = LinearRegression()
        self.initial_embed(reduce_alg, reduce_dim)
        self.initial_classifier(classifier)
    
    def predict(self, X1, y1=None, y_permute=None):
        """
        This function will find the clean set of 
        (X1, y1) and (X2, y2) iteratively.
        """
        # only one set provided, will partition it into two subsets.
        X1 = self.embed(X1)
        n = len(X1)
        q = []
        random_orders = np.random.permutation(n)
        first_half, second_half = random_orders[:n//2], random_orders[n//2:]
        X2_c, y2_c = X1[second_half], y1[second_half]
        X1_c, y1_c = X1[first_half], y1[first_half]
        y1_permute, y2_permute = y_permute[first_half], y_permute[second_half]

        first_prior_set = self.spr(X1_c, y1_c, clean_ratio=0.5) 
        second_prior_set = self.spr(X2_c, y2_c, clean_ratio=0.5) 
        if self.spr_mode == 'spr':
            clean_set = list(np.concatenate([first_half[first_prior_set], second_half[second_prior_set]]))
            q = [0]
        elif self.spr_mode == 'knockoffs':
            first_clean_set, q_1 = self.get_clean_data(X2_c[second_prior_set], y2_c[second_prior_set], X1_c, y1_c, y1_permute)
            q += q_1

            second_clean_set, q_2 = self.get_clean_data(X1_c[first_prior_set], y1_c[first_prior_set], X2_c, y2_c, y2_permute)
            q += q_2

            clean_set = list(np.concatenate([first_half[first_clean_set], second_half[second_clean_set]]))
        return clean_set, q


    def get_clean_data(self, X1, y1, X2=None, y2=None, y2_permute=None):
        # estimation of LR on the first subset
        Y1 = self.label2onehot(y1)
        self.lr.fit(X1, Y1)
        coef, intercept = self.lr.coef_, self.lr.intercept_

        # running on the second subset
        if y2 is None:
            return_y2 = True
            self.classifier.fit(X1, y1)
            y2_probs = self.classifier.predict_proba(X2)
            y2 = y2_probs.argmax(-1)
            for i in range(len(y2_probs)):
                y2_probs[i,y2[i]] = -1 
            y2_permute = y2_probs.argmax(-1)
        else:
            return_y2 = False
            if self.permute_strategy == 'random':
                y2_permute = [self.other_class(c) for c in y2]

        n2 = len(y2)
        X2_concat = np.concatenate([X2, X2])
        y2_concat = np.concatenate([y2, y2_permute])
        Y2_concat = self.label2onehot(y2_concat)
        clean_set, q = self.kspr(X2_concat, Y2_concat, n2, coef, intercept)
        if return_y2:
            return clean_set, y2, q 
        else:
            return clean_set, q
    
    def spr(self, X, label, clean_ratio):
        Y = self.label2onehot(label)
        H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        y_hat = np.dot(X_hat, Y)

        if isinstance(clean_ratio, list):
            clean_num = clean_ratio  
        else: 
            clean_num = []
            for c in range(self.num_class):
                clean_num.append(int(clean_ratio * np.sum(label==c)))

        _, coefs, _ = enet_path(X_hat, y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, :, :]), axis=2)
        Z = coefs.argmin(0)
        clean_set = []
        for c in range(self.num_class):
            c_indexes = np.where(label==c)[0].tolist()
            if len(c_indexes) < clean_num[c]:
                clean_set += c_indexes
            else:
                clean_indexes = Z[c_indexes].argsort()[:clean_num[c]]
                clean_set += np.array(c_indexes)[clean_indexes].tolist()
        return clean_set

    def kspr(self, X, Y, n, coef, intercept):
        Y_hat = Y - np.dot(X, coef.T) - intercept
        X_hat = np.eye(2*n)
        _, coefs, _ = enet_path(X_hat, Y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, :, :]), axis=2)
        Z = coefs.argmin(0)

        W = np.array([Z[i] * np.sign(Z[i]-Z[i+n]) for i in range(n)])
        W_norm_sorted_index = np.abs(W).argsort()
        W_sorted = W[W_norm_sorted_index]
        W_sorted_sign = np.sign(W_sorted)
        positive = np.sum(W_sorted_sign==1)
        negative = np.sum(W_sorted_sign==-1)
        W_sorted_sign = W_sorted_sign.tolist()

        q_list = []
        clean_set = []
        label = Y[:n].argmax(-1)
        for c in range(self.num_class):
            q = 0
            while q < self.threshold:
                q += 0.02
                W_c = deepcopy(W)[label==c]
                W_norm_sorted_index_c = np.abs(W_c).argsort()
                W_sorted_c = W_c[W_norm_sorted_index_c]
                W_sorted_sign_c = np.sign(W_sorted_c)
                positive_c = np.sum(W_sorted_sign_c==1)
                negative_c = np.sum(W_sorted_sign_c==-1)
                W_sorted_sign_c = W_sorted_sign_c.tolist()
                selected_sign_c = deepcopy(W_sorted_sign_c)
                while len(selected_sign_c) > 0:
                    if (1+positive_c) / max(negative_c, 1) <= q:
                        break 
                    else:
                        removed_sign_c = selected_sign_c.pop()
                        while removed_sign_c == -1 and len(selected_sign_c) > 0:
                            negative_c -= 1
                            removed_sign_c = selected_sign_c.pop()
                        if removed_sign_c == 1:
                            positive_c -= 1
                if len(selected_sign_c) > 0:
                    clean_set_c = []
                    for i in range(len(selected_sign_c)):
                        if selected_sign_c[i]==-1:
                            clean_set_c.append(W_norm_sorted_index_c[i])
                    break
            q_list.append(q)
            if q >= self.threshold:
                W_c = deepcopy(W)[label==c]
                W_norm_sorted_index_c = np.abs(W_c).argsort()
                W_sorted_c = W_c[W_norm_sorted_index_c]
                W_sorted_sign_c = np.sign(W_sorted_c)
                positive_c = np.sum(W_sorted_sign_c==1)
                negative_c = np.sum(W_sorted_sign_c==-1)
                W_sorted_sign_c = W_sorted_sign_c.tolist()
                clean_set_c = []
                max_num = len( W_sorted_sign_c)//2
                select_num = 0
                for i in range(len(W_sorted_sign_c)):
                    if W_sorted_sign_c[i]==-1:
                        clean_set_c.append(W_norm_sorted_index_c[i])
                        select_num += 1
                    if select_num == max_num:
                        break
            c_num = 0
            local2gloabel = {}
            for i in range(len(label)):
                if label[i] == c:
                    local2gloabel[c_num] = i 
                    c_num += 1
            for ind in clean_set_c:
                clean_set.append(local2gloabel[ind])  

        return clean_set, q_list

    def initial_embed(self, reduce_alg, d):
        reduce_alg = reduce_alg.lower()
        assert reduce_alg in ['isomap', 'ltsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce_alg == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce_alg == 'ltsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce_alg == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce_alg == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, 
                                           n_neighbors=5,
                                           eigen_solver='dense')
        elif reduce_alg == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce_alg == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce_alg == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm', 'knn']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', 
                                  kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        elif classifier == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier(n_neighbors=1)


    def label2onehot(self, label):
        result = np.zeros((label.shape[0], self.num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result

    def other_class(self, current_class):
        if current_class < 0 or current_class >= self.num_class:
            error_str = "class_ind must be within the range (0, nb_classes - 1)"
            raise ValueError(error_str)

        other_class_list = list(np.arange(self.num_class))
        other_class_list.remove(current_class)
        other_class = np.random.choice(other_class_list)
        return other_class
