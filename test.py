from ekonlpy.sentiment import KSA

text = "진짜 존경한다 어떻게 사람들한테 안좋은것만 저렇게 잘만들까"
ksa = KSA()
tokens = ksa.tokenize(text)
score = ksa.get_score(tokens)
print(score) # {'Positive': 9, 'Negative': -4, 'Polarity': 0.3846153550295881, 'Subjectivity': 0.9999999230769291}

text1 = "전세계를 위협중인 코로나바이러스를 퇴치하기 위해 이렇게까지 노력중인 수많은 연구진과 위료진들께 감사할 따름입니다.. 정말 고맙습니다:)"
tokens1 = ksa.tokenize(text1)
score1 = ksa.get_score(tokens1)
print(score1) # {'Positive': 15, 'Negative': -3, 'Polarity': 0.6666666296296316, 'Subjectivity': 0.9999999444444475}