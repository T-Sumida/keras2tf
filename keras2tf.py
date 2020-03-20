# coding: utf-8
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from keras.applications import resnet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout


def create_keras_model(output_path):
    base = resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(225, 225, 3)
    )

    X = base.output
    X = GlobalAveragePooling2D()(X)
    X = Dropout(0.5)(X)
    X = Dense(2, activation='softmax')(X)
    model = Model(inputs=base.input, outputs=X)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=0.001, clipnorm=1.),
        metrics=['accuracy']
    )
    model.save(output_path)


def freeze_session(session, keep_var_names=None, output_names=None):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


if __name__ == "__main__":
    # create keras model for testing and output h5 file
    create_keras_model("./sample_keras.h5")

    # load keras model
    model = keras.models.load_model("./sample_keras.h5", compile=False)
    print("==========keras input names============")
    print(model.input_names)
    print("=======================================")
    print("==========keras output names===========")
    print(model.output_names)
    print("=======================================")

    # create frozen session
    frozen_graph = freeze_session(
        K.get_session(),
        output_names=[out.op.name for out in model.outputs]
    )

    # output pb
    tf.train.write_graph(frozen_graph, "./", "tfmodel.pb", as_text=False)
    tf.train.write_graph(frozen_graph, './', "tfmodel.pbtxt", as_text=True)
