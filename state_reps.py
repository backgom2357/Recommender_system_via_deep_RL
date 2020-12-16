import tensorflow as tf

class Embedding(tf.keras.Model):
    def __init__(self, users_num, items_num, embedding_dim):
        super(Embedding, self).__init__()
        self.user_eb = tf.keras.layers.Embedding(users_num, embedding_dim)
        self.item_eb = tf.keras.layers.Embedding(items_num, embedding_dim)
        
    def call(self, user_id, items_ids):
        return self.user_eb(user_id), self.item_eb(items_ids)
    
    def _get_item_eb(self, items_ids):
        return self.item_eb(items_ids)

class DRRAveStateRepresentation(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.wav = tf.keras.layers.Conv1D(embedding_dim, 1, 1)
        
    def call(self, user_eb, items_eb):
        wav = self.wav(items_eb)
        user_wav = tf.keras.layers.multiply([user_eb, wav])
        concat = tf.keras.layers.Concatenate([user_eb, user_wav, wav])
        return concat