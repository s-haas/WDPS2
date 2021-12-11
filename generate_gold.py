import os
import shutil

import tensorflow as tf
import pandas as pd

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz", 
                                  origin=URL,
                                  untar=True,
                                  cache_dir='.',
                                  cache_subdir='')

def split_train_test(dataset):
    main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    # Create sub directory path ("/aclImdb/train")
    train_dir = os.path.join(main_dir, 'train')
    # Remove unsup folder since this is a supervised learning task
    rm_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(rm_dir)
    # View the final train folder
    print(os.listdir(train_dir))

train = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=30000, validation_split=0.2, 
    subset='training', seed=128)
test = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=30000, validation_split=0.2, 
    subset='validation', seed=128)


def to_df(raw_data):
    for i in raw_data.take(1):
        feat = i[0].numpy()
        lbl = i[1].numpy()

    df = pd.DataFrame([feat, lbl]).T
    df.columns = ['sentence', 'label']
    df['sentence'] = df['sentence'].str.decode("utf-8")
    return df

def to_input(train, test, feat_col, lbl_col): 
  train_input = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[feat_col], 
                                                          text_b = None,
                                                          label = x[lbl_col]), axis = 1)

  val_input = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[feat_col], 
                                                          text_b = None,
                                                          label = x[lbl_col]), axis = 1)
  
  return train_input, val_input

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'text', 
                                                                           'label')
  
def to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

