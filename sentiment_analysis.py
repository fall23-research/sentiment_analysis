from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle, re
from konlpy.tag import Okt

okt = Okt()

tokenizer_path = 'tokenizer.pickle'
loaded_model = load_model('sentiment_model.h5')

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
MAX_LEN = 60

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩 -> 불러온 토크나이저 사용
  pad_new = pad_sequences(encoded, maxlen = 18) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 코멘트 입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 코멘트 입니다.\n".format((1 - score) * 100))

sentiment_predict('이야.. 다른건 허술하게 만들어놓고서 미세먼지나 코로나는 어쩜저렇게 튼튼하게 만들었을까..')
sentiment_predict('진짜 존경한다 어떻게 사람들한테 안좋은것만 저렇게 잘만들까')
sentiment_predict('전세계를 위협중인 코로나바이러스를 퇴치하기 위해 이렇게까지 노력중인 수많은 연구진과 위료진들께 감사할 따름입니다.. 정말 고맙습니다:)')
sentiment_predict('교수님이 이렇게 강한 코로나 바이러스는 과학 실험실안에서 만들었을 가능성이 높다고 하셨는데 그게 맞지 않을까 싶다.')
sentiment_predict('이제 다시 힘든 상황이 올수도 있을것 같은데 모두 코로나 예방수칙 철저히 잘 지켜서 잘 이겨낼수있었으면 좋겠네요 항상 고생하시는 의료진분들께도 항상 고맙습니다. 마스크를 쓰지 않은채로 예전처럼 일상생활을 할수 있을 날이 왔으면 좋겠어요')