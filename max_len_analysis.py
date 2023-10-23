import numpy as np
import matplotlib.pyplot as plt

# 코멘트 길이 데이터 (예: [10, 20, 30, 40, 50])
comment_lengths = np.array([10, 20, 30, 40, 50])

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

plt.show()
