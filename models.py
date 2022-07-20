from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, Add, Concatenate
from keras.regularizers import L2


class HRNet:
    def __init__(self, input_shape, c):
        self.input_shape = input_shape
        self.c = c

        self.stem_block = self.build_stem()

    def build_bottleneck(self, inputs, in_filters, expansion=4):
        # channel compression
        x = Conv2D(filters=in_filters // expansion, kernel_size=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # convolution
        x = Conv2D(filters=in_filters // expansion, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # expansion
        x = Conv2D(filters=in_filters, kernel_size=1, padding='same')(x)
        x = BatchNormalization()(x)

        # residual
        out = inputs + x
        out = Activation('relu')(out)

        return keras.models.Model(inputs, out)

    def build_basic(self, inputs, out_filters):
        # conv1
        x = Conv2D(filters=out_filters, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # conv2
        x = Conv2D(filters=out_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)

        # residual
        x += inputs
        out = Activation('relu')(x)

        return keras.models.Model(inputs, out)

    def build_stem(self):
        block = keras.models.Sequential([
            Conv2D(filters=self.c, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=self.c, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ], name='stem_block')
        return block

    def build_one_residual_block(self, inputs, in_filters, block):
        if block == 'bottle':
            block = self.build_bottleneck(inputs=inputs, in_filters=in_filters)
        else:
            block = self.build_basic(inputs=inputs, out_filters=in_filters)
        return block

    def build_residual_unit(self, num_blocks, inputs, block_name):
        blocks = []
        for i in range(num_blocks):
            block = self.build_one_residual_block(inputs, inputs.shape[-1], block_name)
            blocks.append(block)
            inputs = block(inputs)   # inputs update
        return keras.models.Sequential(blocks)

    def build_downsample(self, in_r, out_r, name):
        layers = []
        for i in range(out_r - in_r):
            layers.append(Conv2D(filters=2 ** (in_r + i) * self.c, kernel_size=3, strides=2, padding='same'))
            layers.append(BatchNormalization())
            layers.append(Activation('relu'))
        return keras.models.Sequential(layers, name)

    def build_upsample(self, in_r, out_r, name):
        i = in_r - out_r
        block = keras.models.Sequential([
            UpSampling2D((2 ** i, 2 ** i), interpolation='bilinear'),
            Conv2D(filters=2 ** (out_r - 1) * self.c, kernel_size=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ], name)
        return block

    def build_stage(self, inputs, num_units, block, name):
        units = []
        for i in range(num_units):
            unit = self.build_residual_unit(4, inputs, block)
            units.append(unit)
            inputs = unit(inputs)
        return keras.models.Sequential(units, name)

    def build_hrnet(self, head, num_class, weight_decay):
        inputs = Input(self.input_shape)

        # stem
        x = self.stem_block(inputs)

        # stage1
        x1_stage1_out = self.build_stage(x, 1, 'bottle', 'stage1_branch1')(x)

        # split and fusion
        x1 = x1_stage1_out
        x2 = self.build_downsample(1, 2, 'stage1_branch1_to_branch2')(x1)

        # stage2
        x1_stage2_out = self.build_stage(x1, 1, 'basic', 'stage2_branch1')(x1)
        x2_stage2_out = self.build_stage(x2, 1, 'basic', 'stage2_branch2')(x2)

        # split and fusion
        x1 = Add()([x1_stage2_out, self.build_upsample(2, 1, 'stage2_branch2_to_branch1')(x2_stage2_out)])
        x2 = Add()([self.build_downsample(1, 2, 'stage2_branch1_to_branch2')(x1_stage2_out), x2_stage2_out])
        x3 = Add()([self.build_downsample(1, 3, 'stage2_branch1_to_branch3')(x1_stage2_out),
                    self.build_downsample(2, 3, 'stage2_branch2_to_branch3')(x2_stage2_out)])

        # stage3
        x1_stage3_out = self.build_stage(x1, 4, 'basic', 'stage3_branch1')(x1)
        x2_stage3_out = self.build_stage(x2, 4, 'basic', 'stage3_branch2')(x2)
        x3_stage3_out = self.build_stage(x3, 4, 'basic', 'stage3_branch3')(x3)

        # split and fusion
        x1 = Add()([x1_stage3_out, self.build_upsample(2, 1, 'stage3_branch2_to_branch1')(x2_stage3_out),
                    self.build_upsample(3, 1, 'stage3_branch3_to_branch1')(x3_stage3_out)])
        x2 = Add()([self.build_downsample(1, 2, 'stage3_branch1_to_branch2')(x1_stage3_out), x2_stage3_out,
                    self.build_upsample(3, 2, 'stage3_branch3_to_branch2')(x3_stage3_out)])
        x3 = Add()([self.build_downsample(1, 3, 'stage3_branch1_to_branch3')(x1_stage3_out),
                    self.build_downsample(2, 3, 'stage3_branch2_to_branch3')(x2_stage3_out),
                    x3_stage3_out])
        x4 = Add()([self.build_downsample(1, 4, 'stage3_branch1_to_branch4')(x1_stage3_out),
                    self.build_downsample(2, 4, 'stage3_branch2_to_branch4')(x2_stage3_out),
                    self.build_downsample(3, 4, 'stage3_branch3_to_branch4')(x3_stage3_out)])

        # stage4
        x1_stage4_out = self.build_stage(x1, 3, 'basic', 'stage4_branch1')(x1)
        x2_stage4_out = self.build_stage(x2, 3, 'basic', 'stage4_branch2')(x2)
        x3_stage4_out = self.build_stage(x3, 3, 'basic', 'stage4_branch3')(x3)
        x4_stage4_out = self.build_stage(x4, 3, 'basic', 'stage4_branch4')(x4)

        # head
        if head == 'pose':
            x1 = Add()([x1_stage4_out, self.build_upsample(2, 1, 'stage4_branch2_to_branch1')(x2_stage4_out),
                        self.build_upsample(3, 1, 'stage4_branch3_to_branch1')(x3_stage4_out),
                        self.build_upsample(4, 1, 'stage4_branch4_to_branch1')(x4_stage4_out)])
            out = Conv2D(filters=num_class, kernel_size=1, padding='same',
                         activation='sigmoid', name='head')(x1)

            return keras.models.Model(inputs, out, name='pose_estimator')

        elif head == 'segmentation':
            x1 = Add()([x1_stage4_out, self.build_upsample(2, 1, 'stage4_branch2_to_branch1')(x2_stage4_out),
                        self.build_upsample(3, 1, 'stage4_branch3_to_branch1')(x3_stage4_out),
                        self.build_upsample(4, 1, 'stage4_branch4_to_branch1')(x4_stage4_out)])
            x2 = Add()([self.build_downsample(1, 2, 'stage4_branch1_to_branch2')(x1_stage4_out), x2_stage4_out,
                        self.build_upsample(3, 2, 'stage4_branch3_to_branch2')(x3_stage4_out),
                        self.build_upsample(4, 2, 'stage4_branch4_to_branch2')(x4_stage4_out)])
            x3 = Add()([self.build_downsample(1, 3, 'stage4_branch1_to_branch3')(x1_stage4_out),
                        self.build_downsample(2, 3, 'stage4_branch2_to_branch3')(x2_stage4_out),
                        x3_stage4_out, self.build_upsample(4, 3, 'stage4_branch4_to_branch3')(x4_stage4_out)])
            x4 = Add()([self.build_downsample(1, 4, 'stage4_branch1_to_branch4')(x1_stage4_out),
                        self.build_downsample(2, 4, 'stage4_branch2_to_branch4')(x2_stage4_out),
                        self.build_downsample(3, 4, 'stage4_branch3_to_branch4')(x3_stage4_out), x4_stage4_out])

            # upsampling
            x2 = UpSampling2D((2, 2), interpolation='bilinear', name='head_upsample2')(x2)
            x3 = UpSampling2D((4, 4), interpolation='bilinear', name='head_upsample3')(x3)
            x4 = UpSampling2D((8, 8), interpolation='bilinear', name='head_upsample4')(x4)

            out = Concatenate(name='head_concat')([x1, x2, x3, x4])
            out = Conv2D(filters=(self.c + self.c * 2 + self.c * 4 + self. c * 8), kernel_size=1, padding='same',
                         name='head_conv')(out)
            out = BatchNormalization(name='head_bn')(out)
            out = Activation('relu', name='head_relu')(out)

            out = Conv2D(filters=num_class, kernel_size=1, padding='same',
                         kernel_regularizer=L2(weight_decay), bias_regularizer=L2(weight_decay),
                         name='out_conv')(out)
            out = UpSampling2D((4, 4), interpolation='bilinear', name='out_upsample')(out)
            out = Activation('softmax', name='out_activation')(out)
            return keras.models.Model(inputs, out)
        else:
            raise 'Unexpected head type'


if __name__ == '__main__':
    # model = HRNet(input_shape=(256, 192, 3), c=32).build_hrnet(head='pose', num_class=17)
    # print(model.summary())
    # tf.keras.utils.plot_model(model, './pose_estimation/model_plot/pose.png')

    model = HRNet(input_shape=(512, 512, 3), c=32).build_hrnet(head='segmentation', num_class=4, weight_decay=0.0001)
    print(model.summary())
    # tf.keras.utils.plot_model(model, './segmentation/model_plot/segmen.png')
