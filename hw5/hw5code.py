import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    idxs_sort = np.argsort(feature_vector)
    features = feature_vector[idxs_sort]
    targets = target_vector[idxs_sort]
    
    n = target_vector.shape[0]
    n_positive_t_l = np.cumsum(targets)[:-1]
    nl = np.arange(1, n)
    p_1_left = n_positive_t_l / nl
    p_0_left = 1 - p_1_left
    h_left = 1 - p_1_left**2 - p_0_left**2

    nr = n - nl
    p_1_right = (np.sum(targets) - n_positive_t_l) / nr
    
    p_0_right = 1 - p_1_right
    h_right = 1 - p_1_right**2 - p_0_right**2

    q = -nl / n * h_left - nr / n * h_right

    unique_features, unique_counts = np.unique(features, return_counts=True)
    threshold_space =  (unique_features[:-1] + unique_features[1:])/2
    q = q[np.cumsum(unique_counts)[:-1]-1]

    return threshold_space, q, threshold_space[np.argmax(q)], np.max(q)



class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = np.iinfo(np.int32).max if max_depth is None else max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf


    def _fit_node(self, sub_X, sub_y, node, depth):
        depth += 1
            
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return depth
        
        n_node_samples = len(sub_y)
        if (depth >= self._max_depth or
                           n_node_samples < self._min_samples_split or
                           n_node_samples < 2 * self._min_samples_leaf):
            
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return depth
        
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            
            if len(feature_vector) == 1:
                continue

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return depth

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        d1 = self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth)
        d2 = self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth)

        return max(d1, d2)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        
        if (self._feature_types[node['feature_split']] == 'real' and x[node['feature_split']] < node['threshold']) or\
              (self._feature_types[node['feature_split']] == 'categorical' and x[node['feature_split']] in node['categories_split']):
            return self._predict_node(x, node['left_child'])
        return self._predict_node(x, node['right_child'])
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        # return {'feature_types':self._feature_types, 'max_depth':self._max_depth, 'min_samples_split':self._min_samples_split, 'min_samples_leaf':self._min_samples_leaf}
        return {'feature_types':self._feature_types}