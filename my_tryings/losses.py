import tensorflow as tf


def focal_loss(y_true, y_pred):
    """
    compute focal loss
    focal_loss = - alpha * (1-pt)^gamma * log(pt)
    """

    gamma = 2
    alpha = 0.25
    target_confidence = y_true
    pred_confidence = y_pred

    total_positive = tf.reduce_sum(target_confidence)+1 # avoid dividing by zero
    pt = tf.where(target_confidence == 1, pred_confidence, 1 - pred_confidence)

    confidence_loss = - alpha * tf.pow(1-pt, gamma) * tf.log(pt)
    confidence_loss = tf.reduce_sum(confidence_loss) / total_positive

    return confidence_loss


def huber_loss(y_true, y_pred):
    """
    compute smooth L1 loss
    f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
           |x| - 0.5 / sigma / sigma    otherwise
    """

    sigma = 3.0
    true_bbox = y_true
    pred_bbox = y_pred

    # Get all the non zero elements in true_mask
    true_mask = tf.where(true_bbox != 0, tf.ones(tf.shape(true_bbox)), tf.zeros(tf.shape(true_bbox)))
    total_positive = tf.reduce_sum(true_mask[:,:,-1])+1 # avoid dividing by zero

    pred_bbox_filtered = tf.multiply(true_mask, pred_bbox)

    sigma_square = sigma * sigma
    HUBER_DELTA = 1.0 / sigma_square
    x = tf.abs(true_bbox - pred_bbox_filtered)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, x - 0.5 * HUBER_DELTA)

    return tf.reduce_sum(x)/total_positive
