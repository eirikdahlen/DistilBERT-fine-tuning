import pandas as pd
import numpy as np
import time
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

start = time.time()

df = pd.read_csv("frikk_eirik_dataset.csv")

X, y = df['text_document'], df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=df['target'])

X_train = X_train.values.tolist()
for i in range(len(X_train)):
    X_train[i] = str(X_train[i])
X_test = X_test.values.tolist()
for i in range(len(X_test)):
    X_test[i] = str(X_test[i])

labels_dict = {'unrelated': 0, 'pro_ed': 1, 'pro_recovery': 2}

y_train = y_train.values.tolist()
for i in range(len(y_train)):
    y_train[i] = labels_dict[y_train[i]]

y_test = y_test.values.tolist()
for i in range(len(y_test)):
    y_test[i] = labels_dict[y_test[i]]

time_a = time.time() - start

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True)
time_b = time.time() - time_a - start
print(f"Created train encodings, time used {time_b}")
test_encodings = tokenizer(X_test, truncation=True, padding=True)
time_c = time.time() - time_b - time_a - start
print(f"Created val encodings, time used {time_c}")

train_dataset = np.array(list(dict(train_encodings).values()))
test_dataset = np.array(list(dict(test_encodings).values()))

BATCH_SIZE = 16

# Create a callback that saves the model's weights every x epochs
checkpoint_path = "training_ckpt2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

save_model = True

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3, return_dict=True)

if save_model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

    model.fit(
        train_dataset[0],
        np.array(y_train),
        epochs=5,
        batch_size=BATCH_SIZE,
        callbacks=[cp_callback]
        )

else:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

preds = model.predict(test_dataset[0])["logits"]

classes = np.argmax(preds, axis=-1)

score = classification_report(y_test, classes, digits=3)
print(score)

total = time.time()  - start
print(f"Done in: {total}")  
