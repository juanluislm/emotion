import tensorflow as tf

class TFInference:
    def __init__(self, model_path, input_name, output_names):
        # parameters for loading data and images
        self.model_path = model_path
        self.graph = tf.Graph()
        self.graph_def = tf.GraphDef()
        with open(self.model_path, "rb") as f:
            self.graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)
        self.input_name = input_name
        self.input_operation = self.graph.get_operation_by_name(input_name)
        try:
            channel = int(self.input_operation.outputs[0].shape.dims[3])
        except:
            channel = 1
        self.input_shape = (int(self.input_operation.outputs[0].shape.dims[1]),
                            int(self.input_operation.outputs[0].shape.dims[2]),
                            channel)
        self.output_names = []
        self.output_operations = []
        for node_name in output_names:
            self.output_names.append(node_name)
            self.output_operations.append(self.graph.get_operation_by_name(node_name))

        self.sess = tf.Session(graph = self.graph)

    def run(self, image):
        return self.sess.run(
            [operation.outputs[0] for operation in self.output_operations],
            {self.input_operation.outputs[0]: image})
