import numpy as np
from tqdm import tqdm
import tensorflow as tf

def compute_preds(self, sess, model, data_generator):
    y_reals = []
    outputs = []
    output_dim = self._model_kwargs.get('output_dim')
    preds = model.outputs
    labels = model.labels[..., :output_dim]
    loss = self._loss_fn(preds=preds, labels=labels)
    fetches = {
        'loss': loss,
        'mae': loss,
        'outputs': model.outputs,
        'global_step': tf.train.get_or_create_global_step()
    }
    for _, (x, y) in enumerate(data_generator):
        feed_dict = {
            model.inputs: x,
            model.labels: y,
        }
        vals = sess.run(fetches, feed_dict=feed_dict)
        outputs.append(vals['outputs'])
        y_reals.append(y)

    y_preds = np.concatenate(outputs, axis=0)
    y_reals = np.concatenate(np.array(y_reals), axis=0)[..., :output_dim]
    return y_preds, y_reals


def evaluate_multiple(supervisor, sess, data, model, output_filename, category='val', **kwargs):
    # Augmentation pattern, disable all sensors individually
    dis_sensors = range(206) #[189, 200]
    all_preds = []
    # Make global
    # y_truths = []
    scaler = data['scaler']
    loader = data[category + '_loader']
    # y_truths = data['y_' + category][:, :, :, 0]

    y_preds, y_reals = compute_preds(supervisor, sess, model, loader.get_iterator())

    for idx, s in tqdm(enumerate(dis_sensors)):
        augmentation_matrix = np.zeros(207)
        augmentation_matrix[s] = 1

        # Generate augmented datasets
        augmented_dataloader = loader.augment(augmentation_matrix)
        # Do inference
        aug_preds, _ = compute_preds(supervisor, sess, model, augmented_dataloader.get_iterator())
        print('results ', s, 'computed')
        # relative_err = MAPE(normal) - MAPE(augmented)
        #  mae error per sensor before - error per sensor after
        err_rel = (y_reals != 0).astype(np.uint8) * (np.abs(y_preds - y_reals) - np.abs(aug_preds - y_reals))
        # Scale relative_err per 'frame' with softmax, then average over time.
        all_preds.append(err_rel)
        print(err_rel.shape)

    pred_mx = np.stack(all_preds)
    # Aggregate over time
    pred_mx = np.sum(pred_mx, axis=1) / pred_mx.shape[1]
    print(pred_mx.shape)
    np.savez('data/metr-la/results/dcrnn_preds', pred_mx)
