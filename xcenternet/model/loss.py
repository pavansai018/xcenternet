import tensorflow as tf
from typing import Dict, List

def heatmap_focal_loss(outputs: Dict, predictions: List) -> tf.Tensor:
    return focal_loss(outputs['heatmap'], predictions[0])

@tf.function
def focal_loss(hm_true: tf.Tensor, hm_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute Focal Loss for dense heatmap prediction

    Parameters:
    -----------
    hm_true: tf.Tensor
        Ground truth heatmap tensor with values [0,1].
        Shape: (batch_size, height, width, channels).
        Exact value 1.0 denotes a positive center location
    hm_pred: tf.Tensor
        Predicted heamap tensor after activation functioni.
        Shape must match `hm_true`.

    Returns:
    --------
    loss: tf.Tensor
        Scalar tensor representing the focal loss
    """
    pos_mask: tf.Tensor = tf.cast(
        tf.equal(hm_true, 1.0), dtype=tf.float32
    )

    neg_mask: tf.Tensor = tf.cast(
        tf.less(hm_true, 1.0), dtype=tf.float32
    )

    neg_weights: tf.Tensor = tf.math.pow(1.0 - hm_true, 4)

    pos_loss: tf.Tensor = (
        -1.0 * tf.math.log(tf.clip_by_value(hm_pred, 1e-5, 1-1e-5)) 
        * tf.math.pow(1.0 - hm_pred, 2) 
        * pos_mask
    )

    neg_loss: tf.Tensor = (
        -1.0 * tf.math.log(tf.clip_by_value(1.0 - hm_pred, 1e-5, 1-1e-5)) 
        * tf.math.pow(hm_pred, 2) 
        * neg_weights 
        * neg_mask
    )

    num_pos: tf.Tensor = tf.reduce_sum(pos_mask)
    pos_loss: tf.Tensor = tf.reduce_sum(pos_loss)
    neg_loss: tf.Tensor = tf.reduce_sum(neg_loss)

    loss: tf.Tensor = tf.cond(
        tf.greater(num_pos, 0), 
        lambda: (pos_loss + neg_loss) / num_pos, 
        lambda: neg_loss
    )
    return loss
