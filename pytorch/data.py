"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


LOG = logging.getLogger('main')
NO_LABEL = -1




class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        
        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    #tvect = Variable(torch.zeros(nclass))
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0
    
    #print (vect_out)
    vect_out = np.roll(vect_out, -1)

    return vect_out

def change_labels (dataset, data_path, label_filename):
    label_filepath = os.path.join(data_path, label_filename)
    #labels = np.loadtxt(label_filepath, dtype='float64')
    
    lines = [line.rstrip('\n') for line in open(label_filepath)]
    
    labels = []
    for line in lines:
        filename = os.path.join('/home/mittal/.encoding/data/VOCdevkit/VOC2012/SegmentationClassAug/', line)
        filename = filename.replace('jpg', 'png')
        target = Image.open(filename)
        vect = get_label_vector(target, 21)
        labels.append(vect) 
    
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        dataset.imgs[idx] = path, labels[idx]
       
    return dataset, labels 

def relabel_dataset_ml(dataset, labels, percent):
   
    ''' 
    indices = np.arange(len(dataset.imgs))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    select_ind = np.random.choice(indices, int(percent*len(dataset.imgs)), replace=False)
    mask[select_ind]= True

    labeled_indices, unlabeled_indices = indices[mask], indices[~mask]
    '''
    train_ids = pickle.load(open('../train_id.pkl'))

    partial_size = int(percent*len(dataset.imgs))

    labeled_indices = train_ids[:partial_size]
    unlabeled_indices = train_ids[partial_size:]

 
 
    for idx in range(len(dataset.imgs)):
        path , _ = dataset.imgs[idx]    
        if idx in labeled_indices:
            dataset.imgs[idx] = path, labels[idx]
        elif idx in unlabeled_indices:
            for i in range(21):
                labels[idx][i] = NO_LABEL
            dataset.imgs[idx] = path, labels[idx]

     
    #for idx in range(len(dataset.imgs)):
       #print (dataset.imgs[idx])
        #print (dataset.imgs[idx])
    
    #labeled_indices = sorted(set(labeled_indices))
    #unlabeled_indices = sorted(set(unlabeled_indices))
    #labeled_indices = sorted(set(range(len(dataset.imgs))) - set(unlabeled_indices))

    return labeled_indices, unlabeled_indices, dataset
    
def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
