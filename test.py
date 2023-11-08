from ekonlpy.sentiment import KSA

text = "진짜 존경한다 어떻게 사람들한테 안좋은것만 저렇게 잘만들까"
ksa = KSA()
tokens = ksa.tokenize(text)
score = ksa.get_score(tokens)
print(score) # {'Positive': 9, 'Negative': -4, 'Polarity': 0.3846153550295881, 'Subjectivity': 0.9999999230769291}