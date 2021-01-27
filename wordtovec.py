import numpy as np
from collections import defaultdict


#Word2Vec , kelimeleri vektör uzayında ifade etmeye çalışan unsupervised (no labels) ve tahmin temelli(prediction-based) bir modeldir kelimelerin vektör temsilleri oluşturulur.
#2 çeşit alt yöntemi vardır: CBOW(Continous Bag of Words) ve Skip-Gram. CBOW, komşu kelimelerden (bağlam kelimeleri) çıktıyı (hedef kelime) tahmin etmeye çalışırken,  Skip-Gram bir hedef kelimeden bağlam kelimelerini tahmin eder. 
#Bu nedenle, komşu kelimelere bakarak hedef kelimeyi tahmin etmeye çalışabiliriz.


class word2vec():

	def __init__(self):
		self.n = settings['n']
		self.lr = settings['learning_rate']
		self.epochs = settings['epochs']
		self.window = settings['window_size']

	def generate_training_data(self, settings, corpus):
		# unique kelimelerin sözlüğü
		word_counts = defaultdict(int)
		for row in corpus:
			for word in row:
				word_counts[word] += 1
		
		self.v_count = len(word_counts.keys())   #Kelime uzunluğu

		# Generate Lookup Dictionaries (vocab)
		self.words_list = list(word_counts.keys()) #Kelime haznesindeki kelimelerin listesi
		# Kelimelerin indekslerini tutar
		self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
		
		self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

		training_data = []

		# kelimeleri one-hot yapma
		for sentence in corpus:
			sent_len = len(sentence)

			
			for i, word in enumerate(sentence):
				# hedef kelimeyi one-hot encoding çevirme
				w_target = self.word2onehot(sentence[i])

				
				w_context = []

				
				for j in range(i - self.window, i + self.window+1):
					
					if j != i and j <= sent_len-1 and j >= 0:
						
						w_context.append(self.word2onehot(sentence[j]))
						
						
				
				training_data.append([w_target, w_context])

		return np.array(training_data)

	def word2onehot(self, word):
		# Başlangıç vektör
		word_vec = [0 for i in range(0, self.v_count)] 
		# kelime indeksleri
		word_index = self.word_index[word]
		# word lere id atama
		word_vec[word_index] = 1
		return word_vec

	def train(self, training_data):

		self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
		self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))  #Rastgele ağırlıklar atanarak eğitim
		self.loss_history = [0]
		# herbir epoch burada yapılır
		for i in range(self.epochs):
			# Başlangıc loss değeri 0 olarak atanır
			self.loss = 0
			# Herbir cycle de eğitim gerçekleşir
			for w_t, w_c in training_data:
				# İleri yönlü hesaplama
				# w_t = hedef kelime için vektör, w_c = bağlam kelimeleri için vektörler
				y_pred, h, u = self.forward_pass(w_t) #lk eğitim örneğini kullanarak ilk çağımızı eğitmeye başlıyoruz
				
				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				
				# Geri Yönlü hesaplama
				 
				self.backprop(EI, h, w_t) #, işlevi kullanarak ağırlıkları değiştirmemiz gereken ayarlama miktarını hesaplamakiçin geri yayılım işlevini kullanırız
				
				# loss değerinin hesaplanması
				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u))) #- Son olarak, her eğitim örneğini bitirdikten sonra genel kaybı kayıp fonksiyonuna göre hesaplarız. 
				self.loss_history.append(self.loss)
			print('Epoch:', i, "Loss:", self.loss)
		return self.loss_history

	def forward_pass(self, x):
		# RGizli katmanda yapılan işlem
		h = np.dot(x, self.w1)
		# matris çarpımı w2 ile
		u = np.dot(h, self.w2)
		y_c = self.softmax(u)
		return y_c, h, u

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)

	def backprop(self, e, h, x):
		
		dl_dw2 = np.outer(h, e)
		dl_dw1 = np.outer(x, np.dot(self.w2, e.T))	
        #Ağırlıkları güncellemek için, ayarlanacak ağırlıkları ( dl_dw1 ve dl_dw2) öğrenme oranıyla çarpıyoruz ve ardından mevcut ağırlıklardan ( w1ve w2) çıkarıyoruz .
		# Ağırlıklar güncellenir
		self.w1 = self.w1 - (self.lr * dl_dw1)
		self.w2 = self.w2 - (self.lr * dl_dw2)
	# Kelime vektörü oluşturulue
	def word_vec(self, word): #Bir Kelime İçin Vektör Edinme
		w_index = self.word_index[word]
		v_w = self.w1[w_index]
		return v_w

	# Benzer kelimeleri bulur.kelimeler arasındaki kosinüs benzerliğini hesaplar
	def vec_sim(self, word, top_n):
		v_w1 = self.word_vec(word)
		word_sim = {}
     #Giriş vektörü, en yakın kelimeleri döndürür
		for i in range(self.v_count):
			#  Kelime listesindeki her kelime için benzer puanı bulun
			v_w2 = self.w1[i]
			theta_sum = np.dot(v_w1, v_w2)
			theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
			theta = theta_sum / theta_den

			word = self.index_word[i]
			word_sim[word] = theta

		words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

		for word, sim in words_sorted[:top_n]:
			print(word, sim)

def visualize_vector(corpus, w2c):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # tüm kelimelerin embedding vektörlerini numpy-array olarak kaydet.
    word_embedding_list = []
    for word in corpus[0]:
        word_embedding_list.append(w2c.word_vec(word))
    word_embedding_array = np.array(word_embedding_list)
    
    # PCA yöntemiyle görselleştirme yapabiliriz.
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_embedding_array)
    
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(corpus[0]):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()



#####################################################################
settings = {
	'window_size': 2,			# bağlam kelimeleri hedef kelimeye komşu olan kelimelerdir.
	'n': 42,					# kelime gömme boyutları, ayrıca gizli katmanın boyutunu da ifade eder
	'epochs': 200,				# Bu, eğitim dönemlerinin sayısıdır
	'learning_rate': 0.01		# Öğrenme oranı, kayıp gradyanına göre ağırlıklara yapılan ayarlama miktarını kontrol eder.
}

text = "Japonya kaynaklı bir dövüş sanatı ve bir spor dalıdır. Judo, Jujutsu'dan geliştirilmiş ve temel ilkeleri 1882'de Dr. Jigoro Kano tarafından tanımlanmıştır. Judo Japon modern dövüş sanatlarının ilk örneği olmuştur. Gendai Budo (Modern Dövüş Sanatları) geleneksel Japon dövüş sanatları okullarının (Koryu) ilkelerinden geliştirilmiştir."

# Kelimeler küçük harfe çevrilir ve gereksiz boşluklar atılır
corpus = [[word.lower() for word in text.split()]]

# one hot encoding eğitim verilerini oluşturmak için, önce word2vec()nesnesi başlatılır.
w2v = word2vec()
#derlemimizi Word2Vec modelinin eğitim alması için one-hot kodlanmış bir temsile dönüştürmek gerekiyor.
# One hot encoging için hazırlama
training_data = w2v.generate_training_data(settings, corpus)

# Eğitim işlemi
loss= w2v.train(training_data)

# Kelime vektörü oluşturna
word = "spor"
vec = w2v.word_vec(word)
print(word, vec)

# Benzer 3 kelime arama
w2v.vec_sim("spor", 3)
visualize_vector(corpus=corpus, w2c=w2v)
