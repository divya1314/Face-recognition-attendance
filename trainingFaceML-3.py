#lable encoder- categorical data into labled data  like if have 2 files dd,losh then dd become 0,and losh become 1

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC    #support vector classifier algorithm
import pickle

#initilizing of embedding & recognizer
embeddingFile = "output/embeddings.pickle"

#New & Empty at initial
#lable file into pickle file
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])
#fit transfor ->syntax

print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, "wb")
f.write(pickle.dumps(labelEnc))
f.close()
