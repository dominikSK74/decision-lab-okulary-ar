import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.regularizers import l2

# 1. ZMUSZAMY TENSORFLOW DO PRACY W FLOAT32
mixed_precision.set_global_policy('float32')

def model_v3_alfa(input_shape=(640, 640, 3)):
    inputs = layers.Input(shape=input_shape)
    L2_VALUE = 2.5e-4

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
    x = layers.SpatialDropout2D(0.10)(x)
    x_40 = x  # Skala 40x40 (średnie obiekty)

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=2, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters=768, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x_20 = x  # Skala 20x20 (duże obiekty)

    # --- NECK (FPN - Połączenia skrótowe) ---
    p20 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x_20)
    p20_up = layers.UpSampling2D(size=(2, 2))(p20)
    p40_in = layers.Conv2D(256, 1, padding='same', use_bias=False)(x_40)
    concat_40 = layers.Concatenate()([p20_up, p40_in])
    concat_40 = layers.SpatialDropout2D(0.15)(concat_40)

    f_40 = layers.SeparableConv2D(256, 3, padding='same', use_bias=False, depthwise_regularizer=l2(L2_VALUE), pointwise_regularizer=l2(L2_VALUE))(concat_40)
    f_40 = layers.BatchNormalization()(f_40)
    f_40 = layers.ReLU()(f_40)

    p40_up = layers.UpSampling2D(size=(2, 2))(f_40)
    p80_in = layers.Conv2D(128, 1, padding='same', use_bias=False)(x_80)
    concat_80 = layers.Concatenate()([p40_up, p80_in])
    concat_80 = layers.SpatialDropout2D(0.1)(concat_80)

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

# 3. ZBUDOWANIE CZYSTEGO MODELU W FLOAT32 I WGRANIE WAG
model = model_v3_alfa(input_shape=(640, 640, 3))
model.load_weights("dl-model-alfa_21-09-2026_07-53-37-best-50epok.keras")
print("Wagi załadowane do modelu float32!")

# 4. KONWERSJA
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] # Zapisze wagi jako f16 (mniejszy plik), ale operacje to f32!

tflite_model = converter.convert()

with open('yolo_custom.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("Udało się wygenerować yolo_custom.tflite!")