import tensorflow as tf

# Defina o conjunto de dados de treinamento
training_data = [
    ("Oi", "Olá, como posso ajudar?"),
    ("Qual é a sua cor favorita?", "Eu sou um programa de computador, eu não tenho uma cor favorita."),
    ("Onde você mora?", "Eu não moro em lugar algum, eu sou um programa de computador."),
    ("Obrigado", "De nada, sempre estou aqui para ajudar.")
]

# Separe as perguntas e respostas em listas separadas
questions = [pair[0] for pair in training_data]
answers = [pair[1] for pair in training_data]

# Tokenize as perguntas e respostas
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
question_seqs = tokenizer.texts_to_sequences(questions)
answer_seqs = tokenizer.texts_to_sequences(answers)

# Adicione padding para que todas as sequências tenham o mesmo comprimento
max_seq_length = max(len(seq) for seq in question_seqs + answer_seqs)
question_seqs = tf.keras.preprocessing.sequence.pad_sequences(question_seqs, maxlen=max_seq_length, padding='post')
answer_seqs = tf.keras.preprocessing.sequence.pad_sequences(answer_seqs, maxlen=max_seq_length, padding='post')

# Defina o modelo de rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.num_words, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(tokenizer.num_words, activation='softmax')
])

# Compile o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treine o modelo
model.fit(question_seqs, answer_seqs, epochs=100)

# Defina uma função para gerar respostas
def generate_response(question):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = tf.keras.preprocessing.sequence.pad_sequences(question_seq, maxlen=max_seq_length, padding='post')
    predicted_seq = model.predict(question_seq)[0]
    predicted_answer = tokenizer.sequences_to_texts([predicted_seq])[0]
    return predicted_answer

# Use o chatbot
while True:
    question = input("Você: ")
    answer = generate_response(question)
    print("Chatbot: " + answer)