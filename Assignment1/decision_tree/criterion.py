"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def entropy(label_dict):
        total = sum(label_dict.values())
        ent = 0.0
        for c in label_dict.values():
            p = c / total
            ent -= p * math.log2(p)
        return ent
    
    total = len(y)
    left_total = len(l_y)
    right_total = len(r_y)

    before = entropy(all_labels)
    after = (left_total / total) * entropy(left_labels) + (right_total / total) * entropy(right_labels)
    info_gain = before - after
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def split_info(l_y, r_y):
        total = len(l_y) + len(r_y)
        left_total = len(l_y)
        right_total = len(r_y)
        split_info = 0.0
        for part in [left_total, right_total]:
            if part == 0:
                continue
            p = part / total
            split_info -= p * math.log2(p)
        return split_info
    
    split_information = split_info(l_y, r_y)
    if split_information == 0:
        return 0
    
    info_gain /= split_information
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def gini(label_dict):
        total = sum(label_dict.values())
        gini = 1.0
        for c in label_dict.values():
            p = c / total
            gini -= p ** 2
        return gini
    
    total = len(y)
    left_total = len(l_y)
    right_total = len(r_y)

    before = gini(all_labels)
    after = (left_total / total) * gini(left_labels) + (right_total / total) * gini(right_labels)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def error_rate(label_dict):
        total = sum(label_dict.values())
        if total == 0:
            return 0.0
        max_count = max(label_dict.values())
        return 1.0 - (max_count / total)
    
    total = len(y)
    left_total = len(l_y)
    right_total = len(r_y)

    before = error_rate(all_labels)
    after = (left_total / total) * error_rate(left_labels) + (right_total / total) * error_rate(right_labels)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
