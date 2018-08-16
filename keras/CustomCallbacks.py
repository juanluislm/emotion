from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, Callback
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pandas import DataFrame

class ModelCheckpointConfusion(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, generator=None, gestures = [],test_split=(None,None)):
        super(ModelCheckpointConfusion, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.test_data = generator
        self.gestures = gestures
        self.test_split = test_split

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)

                        # self.eval_model(filepath)
                        self.test_model(filepath)


                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def test_model(self, filepath):

        labels = np.zeros( (len(self.test_split[0]), len(self.gestures)) )

        for i in range(0, len(self.test_split[0])  ):

            labels[i] = self.model.predict( self.test_split[0][i] )


    def eval_model(self, base_path):

        labels = self.model.predict_generator(self.test_data, verbose=1)

        confusion_matrix = self.GetConfusion(labels)

        self.SaveToExcel(confusion_matrix, base_path)

    def SaveToExcel(self, confusion, root):

        table_trans = confusion.transpose()

        df = DataFrame({'Label': self.gestures})

        for i in range(0, len(self.gestures)):
            df[self.gestures[i]] = table_trans[i]

        df.to_excel(root + '_confusion.xlsx', sheet_name='sheet1', index=False)
        print(df)

    def GetConfusion(self, labels):

        confusion = np.zeros(  (len(self.gestures), len(self.gestures)), dtype=np.float64  )
        label_count = np.zeros( (len(self.gestures),)  )
        for i in range(0, len(labels)):

            # class_idx = self.test_data.classes[i]
            class_idx = self.test_split[1][i]
            predicted = np.argmax(labels[i])
            # print(predicted, class_idx, labels[i])

            #     print(class_idx, predicted)

            confusion[class_idx][predicted] += 1
            label_count[class_idx] += 1

        for i in range(0, len(self.gestures)):
            for j in range(0, len(self.gestures)):
                confusion[i][j] = confusion[i][j] / label_count[i]

        # predictions =  np.zeros( (len(labels),) )
        # for i in range(0, len(labels)):
        #     predictions[i] = np.argmax(labels[i])
        #
        # confusion = confusion_matrix(test_data.classes, predictions)

        return np.array(confusion)