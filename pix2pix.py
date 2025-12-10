import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. 必要なライブラリのインストール (必要に応じてコメント解除して実行)
# !pip install tensorflow
# !pip install matplotlib

# --- 2. データセットの準備
# Pix2pixは入力画像と出力画像のペアを必要とします。
# 例: 白黒画像 -> カラー画像、航空写真 -> 地図
# ここではダミーデータを例として示します。

# データセットのパスやロード関数を定義します。
# def load_image_pair(input_path, target_path):
#     # 画像の読み込み、リサイズ、正規化などの前処理を行います。
#     input_image = tf.io.read_file(input_path)
#     input_image = tf.image.decode_jpeg(input_image, channels=3)
#     input_image = tf.image.resize(input_image, [256, 256])
#     input_image = (input_image / 127.5) - 1 # Normalize to [-1, 1]

#     target_image = tf.io.read_file(target_path)
#     target_image = tf.image.decode_jpeg(target_image, channels=3)
#     target_image = tf.image.resize(target_image, [256, 256])
#     target_image = (target_image / 127.5) - 1 # Normalize to [-1, 1]

#     return input_image, target_image

# 仮のデータセット作成 (実際のPix2pixでは画像ファイルからロードします)
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def generate_random_data(num_samples):
    # 実際の画像データに置き換えてください
    inputs = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32) * 2 - 1
    targets = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32) * 2 - 1
    return tf.data.Dataset.from_tensor_slices((inputs, targets))

train_dataset = generate_random_data(100).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = generate_random_data(10).batch(BATCH_SIZE)


# --- 3. Pix2pixモデルの構築
# Pix2pixはGenerator (U-Netライク) と Discriminator (PatchGAN) で構成されます。

# Generatorの定義 (U-Netスタイル)
def unet_generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4),
        upsample(512, 4),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        filters=3, kernel_size=4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'
    )

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1]) # Skip the last encoder output

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

# Discriminatorの定義 (PatchGANスタイル)
def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# モデルのインスタンス化
generator = unet_generator()
discriminator_model = discriminator()

# --- 4. 損失関数とオプティマイザの定義
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# --- 5. 学習ループの定義
EPOCHS = 1

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator_model([input_image, target], training=True)
        disc_generated_output = discriminator_model([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_variables))

# 学習の実行
print("\n--- Training Pix2pix Model ---")
for epoch in range(EPOCHS):
    # 実際のデータセットでループ
    for n, (input_image, target) in enumerate(train_dataset.take(10)):
        train_step(input_image, target, epoch)
        if n % 10 == 0:
            print(f".", end='')
    print(f"\nEpoch {epoch+1} finished")


# --- 6. 推論の実行
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


# テストデータで推論を試す
print("\n--- Generating Sample Images ---")
for inp, tar in test_dataset.take(1):
    generate_images(generator, inp, tar)


print("Pix2pixの基本的な実行フローが完了しました。")
print("ご自身のデータセットとモデルのハイパーパラメータに合わせて調整してください。")
