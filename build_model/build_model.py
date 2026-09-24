import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from sklearn.cluster import KMeans
import os
from datetime import datetime

mixed_precision.set_global_policy('mixed_float16')

print("Wersja TF:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def create_model(input_shape=(640, 640, 3)):
    inputs = layers.Input(shape=input_shape)

    L2_VALUE = 1e-4

    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=l2(L2_VALUE))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x_40 = x
    
    x = layers.SeparableConv2D(filters=384, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.SeparableConv2D(filters=384, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x_20 = x
    
    #Siatka dla 20x20
    out_large = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_large')(x_20)
    out_large = layers.Reshape((20, 20, 3, 8), name='out_large')(out_large)

    #Siatka dla 40x40
    out_small = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_small')(x_40)
    out_small = layers.Reshape((40, 40, 3, 8), name='out_small')(out_small)
    
    model = models.Model(inputs=inputs, outputs=[out_large, out_small], name="dl-model-alfa-1")
    return model

def model_v2_alfa(input_shape=(640, 640, 3)):
    inputs = layers.Input(shape=input_shape)

    L2_VALUE = 1e-4

    x = layers.Conv2D(filters=48, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=l2(L2_VALUE))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=96, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=96, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=192, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=192, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=320, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=320, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x_80 = x

    x = layers.SeparableConv2D(filters=480, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=480, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x_40 = x

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x_20 = x

    # siatka dla 20x20
    out_large = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_large')(x_20)
    out_large = layers.Reshape((20, 20, 3, 8), name='out_large')(out_large)

    # siatka dla 40x40
    out_medium = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_medium')(x_40)
    out_medium = layers.Reshape((40, 40, 3, 8), name='out_medium')(out_medium)

    # siatka dla 80x80
    out_small = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_small')(x_80)
    out_small = layers.Reshape((80, 80, 3, 8), name='out_small')(out_small)

    model = models.Model(inputs=inputs, outputs=[out_large, out_medium, out_small], name="dl-model-alfa-2")
    return model


def model_v3_alfa(input_shape=(640, 640, 3)):
    inputs = layers.Input(shape=input_shape)
    L2_VALUE = 1e-4

    # --- BACKBONE (Szkielet) ---
    x = layers.Conv2D(filters=48, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=l2(L2_VALUE))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=96, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=96, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=192, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=192, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=320, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=320, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x_80 = x  # Skala 80x80 (małe obiekty)

    x = layers.SeparableConv2D(filters=480, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=480, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x_40 = x  # Skala 40x40 (średnie obiekty)

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x_20 = x  # Skala 20x20 (duże obiekty)

    # --- NECK (FPN - Połączenia skrótowe) ---
    # 1. Przeniesienie informacji z 20x20 do 40x40
    p20 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x_20)
    p20_up = layers.UpSampling2D(size=(2, 2))(p20)
    p40_in = layers.Conv2D(256, 1, padding='same', use_bias=False)(x_40)
    concat_40 = layers.Concatenate()([p20_up, p40_in])
    
    f_40 = layers.SeparableConv2D(256, 3, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(concat_40)
    f_40 = layers.BatchNormalization()(f_40)
    f_40 = layers.ReLU()(f_40)

    # 2. Przeniesienie informacji z 40x40 do 80x80
    p40_up = layers.UpSampling2D(size=(2, 2))(f_40)
    p80_in = layers.Conv2D(128, 1, padding='same', use_bias=False)(x_80)
    concat_80 = layers.Concatenate()([p40_up, p80_in])
    
    f_80 = layers.SeparableConv2D(128, 3, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(concat_80)
    f_80 = layers.BatchNormalization()(f_80)
    f_80 = layers.ReLU()(f_80)

    # --- HEAD (Głowa detekcyjna) ---
    f_20 = layers.SeparableConv2D(512, 3, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x_20)
    f_20 = layers.BatchNormalization()(f_20)
    f_20 = layers.ReLU()(f_20)

    out_large = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_large')(f_20)
    out_large = layers.Reshape((20, 20, 3, 8), name='out_large')(out_large)

    out_medium = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_medium')(f_40)
    out_medium = layers.Reshape((40, 40, 3, 8), name='out_medium')(out_medium)

    out_small = layers.Conv2D(24, 1, activation='linear', dtype='float32', name='conv_small')(f_80)
    out_small = layers.Reshape((80, 80, 3, 8), name='out_small')(out_small)

    model = models.Model(inputs=inputs, outputs=[out_large, out_medium, out_small], name="dl-model-alfa-fpn")
    return model


# Liczebność danych
# Klasa 0: 10792
# Klasa 1: 9241
# Klasa 2: 5870
CLASS_COUNTS = tf.constant([10792.0, 9241.0, 5870.0])
CLASS_WEIGHTS = tf.reduce_sum(CLASS_COUNTS) / (3.0 * CLASS_COUNTS)

# def loss_function(y_true, y_pred):
#     """
#     inputs: y_true and y_pred: (Batch, 20, 20, 3, 8)
#     return: total loss
#     """
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     # masks
#     object_mask = y_true[..., 4] 
#     no_object_mask = 1.0 - object_mask

#     num_objects = tf.reduce_sum(object_mask) + 1e-6
#     num_background = tf.reduce_sum(no_object_mask) + 1e-6
    
#     # loss of location
#     # pred_box = y_pred[..., 0:4] # x, y, width, height
#     # true_box = y_true[..., 0:4] # x, y, width, height
#     #MSE = tf.square(true_box - pred_box)
#     #sum_MSE = tf.reduce_sum(MSE, axis = -1)
#     #location_loss = 5.0 * tf.reduce_sum(object_mask * sum_MSE) / num_objects # 5.0 is a weight
#     #=====
#     pred_xy = tf.math.sigmoid(y_pred[..., 0:2])
#     pred_wh = tf.math.exp(tf.clip_by_value(y_pred[..., 2:4], -5.0, 5.0))
#     pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
#     true_box = y_true[..., 0:4]

#     true_xy, true_wh = true_box[..., 0:2], true_box[..., 2:4]
#     loss_xy = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
#     loss_wh = tf.reduce_sum(tf.square(tf.sqrt(tf.maximum(true_wh, 1e-7)) - tf.sqrt(pred_wh)), axis=-1)

#     location_loss = 5.0 * tf.reduce_sum(object_mask * (loss_xy + loss_wh)) / num_objects
#     #=====

#     # loss of confidence (objects)
#     sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#         labels = y_true[..., 4],
#         logits = y_pred[..., 4]
#     )
#     confidence_loss_objects = tf.reduce_sum(object_mask * sigmoid_cross_entropy) / num_objects

#     # loss of confidence (background)
#     confidence_loss_background = 0.5 * tf.reduce_sum(no_object_mask * sigmoid_cross_entropy) / num_background # 0.5 is weight

#     # classification loss (3 classes, indexes: 5, 6, 7 OneHot code)
#     true_classes = y_true[..., 5:8]
#     pred_classes = y_pred[..., 5:8]

#     softmax = tf.nn.softmax_cross_entropy_with_logits(
#         labels = true_classes,
#         logits = pred_classes
#     )
#     sample_weights = tf.reduce_sum(true_classes * CLASS_WEIGHTS, axis=-1)
#     classification_loss = tf.reduce_sum(object_mask * softmax* sample_weights)  / num_objects

#     # total loss
#     loss = location_loss + confidence_loss_objects + confidence_loss_background + classification_loss
#     # total_loss = tf.reduce_mean(loss)
#     # total_loss = loss / tf.cast(tf.shape(y_true)[0], tf.float32)
#     return loss

# def loss_function(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
    
#     # Maski
#     object_mask = y_true[..., 4] 
#     no_object_mask = 1.0 - object_mask

#     # Liczba obiektów i komórek tła z epsilonem (aby uniknąć dzielenia przez zero)
#     num_objects = tf.reduce_sum(object_mask) + 1e-6
#     num_background = tf.reduce_sum(no_object_mask) + 1e-6

#     # Bezpieczne koordynaty (wymuszamy wartości 0.0 - 1.0)
#     pred_xy = tf.math.sigmoid(y_pred[..., 0:2])
#     pred_wh = tf.math.sigmoid(y_pred[..., 2:4])
    
#     true_xy = y_true[..., 0:2]
#     true_wh = y_true[..., 2:4]

#     # Strata lokalizacji z zabezpieczeniem pierwiastka
#     loss_xy = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
#     loss_wh = tf.reduce_sum(tf.square(tf.sqrt(tf.maximum(true_wh, 1e-7)) - tf.sqrt(tf.maximum(pred_wh, 1e-7))), axis=-1)
    
#     # DZIELIMY TYLKO PRZEZ LICZBĘ OBIEKTÓW
#     location_loss = 5.0 * tf.reduce_sum(object_mask * (loss_xy + loss_wh)) / num_objects

#     # Strata pewności (Confidence)
#     sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#         labels=y_true[..., 4],
#         logits=y_pred[..., 4]
#     )
#     # DZIELIMY OSOBNO PRZEZ OBIEKTY I TŁO
#     confidence_loss_objects = tf.reduce_sum(object_mask * sigmoid_cross_entropy) / num_objects
#     confidence_loss_background = 0.5 * tf.reduce_sum(no_object_mask * sigmoid_cross_entropy) / num_background

#     # Strata klasyfikacji (Softmax i One-Hot)
#     true_classes = y_true[..., 5:8]
#     pred_classes = y_pred[..., 5:8]

#     softmax = tf.nn.softmax_cross_entropy_with_logits(
#         labels=true_classes,
#         logits=pred_classes
#     )
#     sample_weights = tf.reduce_sum(true_classes * CLASS_WEIGHTS, axis=-1)
#     # DZIELIMY TYLKO PRZEZ LICZBĘ OBIEKTÓW
#     classification_loss = tf.reduce_sum(object_mask * softmax * sample_weights) / num_objects

#     # Suma całkowita - poszczególne składowe są już znormalizowane
#     total_loss = location_loss + confidence_loss_objects + confidence_loss_background + classification_loss
    
#     return total_loss


def create_loss_function(anchors):
    # Kształtujemy anchory do postaci (1, 1, 1, 3, 2), 
    # aby automatycznie "rozmnożyły się" na wymiary (Batch, Grid_Y, Grid_X, 3, 2)
    anchors_tensor = tf.constant(anchors, dtype=tf.float32)
    anchors_tensor = tf.reshape(anchors_tensor, (1, 1, 1, len(anchors), 2))

    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Maski
        object_mask = y_true[..., 4] 
        no_object_mask = 1.0 - object_mask

        # Liczba obiektów i tła
        num_objects = tf.reduce_sum(object_mask) + 1e-6
        num_background = tf.reduce_sum(no_object_mask) + 1e-6

        # --- TUTAJ JEST MAGIA ANCHORÓW ---
        pred_xy = tf.math.sigmoid(y_pred[..., 0:2])
        
        # Wzór z YOLOv5: Zamiast tf.exp używamy (2 * sigmoid)^2 * anchor
        # To stabilizuje trening (gradienty nie wybuchają) i pozwala sieci
        # na skalowanie anchora od 0 do 4x jego wielkości.
        pred_wh = (tf.math.sigmoid(y_pred[..., 2:4]) * 2.0) ** 2 * anchors_tensor
        # ---------------------------------
        
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]

        # Strata lokalizacji z zabezpieczeniem pierwiastka
        loss_xy = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        loss_wh = tf.reduce_sum(tf.square(tf.sqrt(tf.maximum(true_wh, 1e-7)) - tf.sqrt(tf.maximum(pred_wh, 1e-7))), axis=-1)
        
        location_loss = 5.0 * tf.reduce_sum(object_mask * (loss_xy + loss_wh)) / num_objects

        # Strata pewności (Confidence)
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true[..., 4],
            logits=y_pred[..., 4]
        )
        confidence_loss_objects = tf.reduce_sum(object_mask * sigmoid_cross_entropy) / num_objects
        confidence_loss_background = 0.5 * tf.reduce_sum(no_object_mask * sigmoid_cross_entropy) / num_background

        # Strata klasyfikacji
        true_classes = y_true[..., 5:8]
        pred_classes = y_pred[..., 5:8]

        softmax = tf.nn.softmax_cross_entropy_with_logits(
            labels=true_classes,
            logits=pred_classes
        )
        sample_weights = tf.reduce_sum(true_classes * CLASS_WEIGHTS, axis=-1)
        classification_loss = tf.reduce_sum(object_mask * softmax * sample_weights) / num_objects

        # Suma całkowita
        total_loss = location_loss + confidence_loss_objects + confidence_loss_background + classification_loss
        
        return total_loss
        
    return loss_function

def select_anchor(anchor_list, width, height):
    """
    input: anchors_list example: [[0.05, 0.05], [0.15, 0.30], [0.40, 0.40]], width and height of object
    return: index the best anchor
    
    Function calculate Intersection Over Union
    """
    IOU = []
    object_field = width * height
    for anchor in anchor_list:
        anchor_field = anchor[0] * anchor[1]
        common_width = min(width, anchor[0])
        common_height = min(height, anchor[1])
        common_field = common_width * common_height
        total_field = object_field + anchor_field - common_field
        IOU.append(common_field / total_field)
    
    return IOU.index(max(IOU))

def prepare_labels(label_path, do_flip):
    """
    input: label path
    return: correctly filled-in matrix (20, 20, 3, 8)
    """

    # matrix = np.zeros((20, 20, 3, 8))
    matrix_large = np.zeros((20, 20, 3, 8))
    matrix_medium = np.zeros((40, 40, 3, 8))
    matrix_small = np.zeros((80, 80, 3, 8))

    if hasattr(label_path, 'numpy'):
        label_path = label_path.numpy()

    if isinstance(label_path, bytes):
        label_path = label_path.decode('utf-8')
    elif not isinstance(label_path, str):
        label_path = str(label_path)

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        values = line.split(' ')
        id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        # relative cords
        if do_flip:
            x_center = 1.0 - x_center

        # OneHot code
        one_hot = [0, 0, 0]
        one_hot[id] = 1

        best_anchor_index = select_anchor(ANCHORS_ALL, width, height)

        if best_anchor_index >= 6:
            grid_size = 20
            anchor_idx = best_anchor_index - 6
            target_matrix = matrix_large
        elif best_anchor_index >= 3:
            grid_size = 40
            anchor_idx = best_anchor_index - 3
            target_matrix = matrix_medium
        else:
            grid_size = 80
            anchor_idx = best_anchor_index
            target_matrix = matrix_small

        x = x_center * grid_size
        y = y_center * grid_size
        x_grid = int(np.floor(x))
        y_grid = int(np.floor(y))
        relative_x = x - x_grid
        relative_y = y - y_grid
        target_matrix[y_grid, x_grid, anchor_idx] = [relative_x, relative_y, width, height, 1] + one_hot

    return matrix_large, matrix_medium, matrix_small

def load_data(img_path, label_path, training = True):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img.set_shape([640, 640, 3])
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        do_flip = tf.random.uniform([], 0, 1) > 0.5
        img = augument(img, do_flip)
    else:
        do_flip = tf.constant(False)

    target_large, target_medium, target_small = tf.py_function(
        func=prepare_labels,
        inp=[label_path, do_flip],
        Tout=[tf.float64, tf.float64, tf.float64]
    )
    target_large.set_shape([20, 20, 3, 8])
    target_medium.set_shape([40, 40, 3, 8])
    target_small.set_shape([80, 80, 3, 8])
    return img, (target_large, target_medium, target_small)



def augument(img, do_flip):
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.02)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def compute_anchors(label_files, n_anchors=3):
    widths_heights = []
    for label_path in label_files:
        with open(label_path, "r") as f:
            for line in f:
                values = line.split(' ')
                widths_heights.append([float(values[3]), float(values[4])])
    
    wh = np.array(widths_heights)
    kmeans = KMeans(n_clusters=n_anchors, n_init=10, random_state=42).fit(wh)
    anchors = sorted(kmeans.cluster_centers_.tolist(), key=lambda a: a[0] * a[1])
    return anchors

TRAIN_IMG_DIR = "data/data2/datasets/dataset-train/images"
TRAIN_LABELS_DIR = "data/data2/datasets/dataset-train/labels"
VALIDATION_IMG_DIR = "data/data2/datasets/dataset-validation/images"
VALIDATION_LABELS_DIR = "data/data2/datasets/dataset-validation/labels"
# TRAIN_IMG_DIR = "/workspace/datasets/datasets/dataset-train/images"
# TRAIN_LABELS_DIR = "/workspace/datasets/datasets/dataset-train/labels"
# VALIDATION_IMG_DIR = "/workspace/datasets/datasets/dataset-validation/images"
# VALIDATION_LABELS_DIR = "/workspace/datasets/datasets/dataset-validation/labels"

train_img_files = sorted([os.path.join(TRAIN_IMG_DIR, f) for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.jpg')])
train_label_files = sorted([os.path.join(TRAIN_LABELS_DIR, f) for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')])
ANCHORS_ALL = compute_anchors(train_label_files, n_anchors=9)
ANCHORS_SMALL = ANCHORS_ALL[0:3] # 80x80
ANCHORS_MEDIUM = ANCHORS_ALL[3:6] #40x40
ANCHORS_LARGE = ANCHORS_ALL[6:9] #20x20
print("Wyliczone anchory:", ANCHORS_ALL)


val_img_files = sorted([os.path.join(VALIDATION_IMG_DIR, f) for f in os.listdir(VALIDATION_IMG_DIR) if f.endswith('.jpg')])
val_label_files = sorted([os.path.join(VALIDATION_LABELS_DIR, f) for f in os.listdir(VALIDATION_LABELS_DIR) if f.endswith('.txt')])

BATCH_SIZE = 8
train_path_ds = tf.data.Dataset.from_tensor_slices((train_img_files, train_label_files))
# train_dataset = train_path_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

# train_dataset = train_dataset.shuffle(buffer_size=15000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_dataset = train_path_ds.shuffle(buffer_size=15000, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(lambda i, l: load_data(i, l, training=True), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


val_path_ds = tf.data.Dataset.from_tensor_slices((val_img_files, val_label_files))
# val_dataset = val_path_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_path_ds.map(lambda i, l: load_data(i, l, training=False), num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# model = create_model(input_shape=(640, 640, 3))
model = model_v3_alfa(input_shape=(640, 640, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'out_large': create_loss_function(ANCHORS_LARGE),
        'out_medium': create_loss_function(ANCHORS_MEDIUM),
        'out_small': create_loss_function(ANCHORS_SMALL)
    },
    loss_weights={
        'out_large': 1.0,
        'out_medium': 1.0,
        'out_small': 1.0
    }
)
model.summary()

# model = tf.keras.models.load_model(
#     "dl-model-alfa-23-07-26-16epoch-1.keras",
#     custom_objects={"loss_function": loss_function}
# )

checkpoint_path_best = datetime.now().strftime("dl-model-alfa_%d-%m-%Y_%H-%M-%S-best.keras")
checkpoint_path_last = datetime.now().strftime("dl-model-alfa_%d-%m-%Y_%H-%M-%S-last.keras")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_best,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

checkpoint_last = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_last,
    save_best_only=False,
    save_freq='epoch',
    verbose=0
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

log_filename = datetime.now().strftime("training_log_%d-%m-%Y_%H-%M-%S.csv")
log_path = os.path.join("logs", log_filename)
csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=True)

print("\n[INFO] Rozpoczynam uczenie sieci...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    #initial_epoch=16,
    callbacks=[checkpoint_callback, checkpoint_last, early_stopping, reduce_lr, csv_logger]
)
