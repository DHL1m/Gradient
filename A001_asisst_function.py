import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import struct

def quantize (W):
    Wb = W * 0
    fff = 0.0625 * 0.125

    # Wb = W

    # Wb = Wb + tf.cast(fff*2.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(fff < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(0.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(0.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff*2.0 > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff*3.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(fff*3.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(-fff*4.0 >= W, tf.float32) * (W +4.0* fff)
    # Wb = Wb + tf.cast(fff*4.0 < W, tf.float32) * (W -4.0* fff)
    # Wb = Wb + tf.cast(-fff*5.0 >= W, tf.float32) * (-W -5.0* fff)
    # Wb = Wb + tf.cast(fff*5.0 < W, tf.float32) * (-W +5.0* fff)


    # Wb = Wb + tf.cast(fff * 2.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(fff < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(0.0 < W, tf.float32)*0
    # Wb = Wb + tf.cast(0.0 >= W, tf.float32)*0
    # Wb = Wb + tf.cast(-fff > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff * 2.0 > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff * 3.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(fff * 3.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(-fff * 4.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(fff * 4.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(-fff*5.0 >= W, tf.float32) * (W +4.0* fff)
    # Wb = Wb + tf.cast(fff*5.0 < W, tf.float32) * (W -4.0* fff)
    # Wb = Wb + tf.cast(-fff*6.0 >= W, tf.float32) * (-W -5.0* fff)
    # Wb = Wb + tf.cast(fff*6.0 < W, tf.float32) * (-W +5.0* fff)

    # Wb = (tf.cast(W > 0, tf.float32)-0.5)*2.0*fff

    # Wb = Wb + (tf.cast(W > fff, tf.float32)) * 3 * fff
    # Wb = Wb + (tf.cast(W < -fff, tf.float32)) * 3 *(-fff)

    # Wb = Wb + tf.cast(fff < W, tf.float32) * 2*(fff)
    # Wb = Wb + tf.cast(0.0 < W, tf.float32) * 2*(fff)
    # Wb = Wb + tf.cast(0.0 >= W, tf.float32) * 2*(-fff)
    # Wb = Wb + tf.cast(-fff > W, tf.float32) * 2*(-fff)


    Wb = Wb + tf.cast(W//fff+1,dtype=tf.float32)*fff
    Wb = Wb + tf.cast(W <0 , dtype=tf.float32)*(-1.0)*fff
    Wb = tf.clip_by_value(Wb,-128*fff,128*fff)

    # Wb = Wb + tf.cast(fff * 2.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(fff < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(0.0 < W, tf.float32) * (fff)
    # Wb = Wb + tf.cast(0.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff * 2.0 > W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(-fff * 3.0 >= W, tf.float32) * (-fff)
    # Wb = Wb + tf.cast(fff * 3.0 < W, tf.float32) * (fff)

    return tf.stop_gradient(Wb - W) + W

# def comp_bit(t):
#     out=0.
#     fff = 0.0625 * 0.125
#     if t > fff:
#         out=1.
#     return out

def Q_operation_Wb(W1b_pre, W1b):
    D1 = np.absolute(W1b_pre - W1b)
    # D1 = D1.reshape(np.size(D1))
    # DD1 = lambda t: comp_bit(t)
    # DDD1 = np.array([DD1(xi) for xi in D1])
    fff = 0.0625 * 0.125
    DDD1 = D1>fff
    return DDD1.sum()

# def Q_operation_W2b(Wb_pre, Wb):
#     D = np.absolute(Wb_pre - Wb)
#     D = D.reshape(100)
#     DD = lambda t: comp_bit(t)
#     DDD = np.array([DD(xi) for xi in D])
#     return DDD.sum()

# def bit_1_count(int_type):
#     # length = 0
# #     count = 0
# #     while (int_type):
# #         count += (int_type & 1)
# #         # length += 1
# #         int_type >>= 1
# # #     return(length, count)
#     count=bin(int_type).count('1')
#     return (count)

# def float32_to_bintext(num):
#
#     binNum = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
#     return binNum.rjust(32, "0")

def FP_operation_W(A,B):
    # A=A.reshape(np.size(A))
    # B=B.reshape(np.size(B))
    AA = A.view(np.uint32)
    BB = B.view(np.uint32)

    # squarer = lambda t: int(float32_to_bintext(t),2)
    # squares1 = np.array([squarer(xi) for xi in A])
    # squares2 = np.array([squarer(xi) for xi in B])
    # AA = hex(struct.unpack('<I', struct.pack('<f', A))[0])
    # BB = hex(struct.unpack('<I', struct.pack('<f', B))[0])
    # squares.reshape(784,10)
    # C=squares1^squares2

    # C=AA^BB
    # squarer2 = lambda t: bit_1_count(t)
    # squares3 = np.array([squarer2(xi) for xi in C])
    # squares3 = bit_1_count(C)


    # return squares3.sum()
    return sum(bin(x).count("1") for x in (AA^BB).reshape(np.size(AA)))


# def FP_operation_W2(A,B):
#     A=A.reshape(100)
#     B=B.reshape(100)
#     squarer = lambda t: int(float32_to_bintext(t),2)
#     squares1 = np.array([squarer(xi) for xi in A])
#     squares2 = np.array([squarer(xi) for xi in B])
#
#     # squares.reshape(784,10)
#     C=squares1^squares2
#     squarer2 = lambda t: bit_1_count(t)
#     squares3 = np.array([squarer2(xi) for xi in C])
#     return squares3.sum()