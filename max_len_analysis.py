import numpy as np
import matplotlib.pyplot as plt

import sqlite3
database = sqlite3.connect("/Users/siwon/Desktop/Fall-23/research/official_news_dataset/news_comment.sqlite")
databaseCursor = database.cursor()

databaseCursor.execute('''SELECT LENGTH(text_display) AS text_length FROM news_comment''')
rows = databaseCursor.fetchall()
length_comments = [row[0] for row in rows]

# 코멘트 길이 데이터 (예: [10, 20, 30, 40, 50])
comment_lengths = np.array(length_comments)

# 정렬
sorted_lengths = np.sort(comment_lengths)

# 누적 개수 계산
cumulative_counts = np.arange(1, len(sorted_lengths) + 1)

# 누적 비율 계산 (CDF)
cumulative_percentages = cumulative_counts / len(sorted_lengths)

# CDF 그래프 그리기
plt.plot(sorted_lengths, cumulative_percentages)
plt.xlabel('Comment Length')
plt.ylabel('Cumulative Percentage')
plt.grid()

# 95% 이상 커버되는 길이 찾기
desired_coverage = 0.95
index = np.argmax(cumulative_percentages >= desired_coverage)
desired_length = sorted_lengths[index]

print(f"95% 이상을 커버하는 max length: {desired_length}")

print('리뷰의 최대 길이 :', sorted_lengths[-1])
print('리뷰의 평균 길이 :', np.mean(length_comments))
plt.hist(length_comments, bins=10, range=(0, 200))
plt.xlabel('Length of Comment')
plt.ylabel('Frequency')
plt.title('Comment Length Distribution (0-200)')
plt.grid(True)
plt.show()

# 95% 이상을 커버하는 max length: 138
# 리뷰의 최대 길이 : 6953
# 리뷰의 평균 길이 : 49.663236238695795
