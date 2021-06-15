import tensorflow as tf


def wgan_d_loss(r_logit, f_logit):
    r_loss = -tf.reduce_mean(r_logit)
    f_loss = tf.reduce_mean(f_logit)
    return r_loss, f_loss


def wgan_g_loss(f_logit):
    f_loss = -tf.reduce_mean(f_logit)
    return f_loss