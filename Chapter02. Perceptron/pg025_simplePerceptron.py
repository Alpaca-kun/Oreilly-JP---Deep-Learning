def AND (x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7 # Attribute values of weight 1, 2 and theta
    tmp = x1 * w1 + x2 * w2

    if tmp <= theta:
        return 0
    else:
        return 1
