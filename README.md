# 연구목표 
● Whole Slide Image 내 공간적 정보 보존을 위한 attention 기반 pooling 방법 고안  
● Mean pooling 대비 제안 기법의 feature 표현력 개선 검증  
● 기존 머신러닝 기법 대비 신경망 기반 생존예측 성능 비교  

# Figures
그림 1. survival attention heatmap
<img width="360" height="360" alt="survival_attention_heatmap" src="https://github.com/user-attachments/assets/6996d1e6-31f8-4ca2-9a2c-9e9ed6ea0d9d" />

(빨강 = 모델이 생존 예측에 중요하게 본 영역  
파랑 = 거의 영향 없는 영역  
노랑/주황 = 중간 기여)  
attention heatmap을 통해 모델의 집중 영역을 시각화하였다. 그러나 해당 영역이 병리학적으로 어떤 조직학적 특성과 대응되는지에 대한 전문적 검증은 수행되지 않았다. 향후 병리 전문가를 통해 모델을 정량적으로 평가할 필요가 있다.

그림 2. 전체 구조도 
<img width="2816" height="1536" alt="Gemini_Generated_Image_cysjcicysjcicysj" src="https://github.com/user-attachments/assets/50dee8f2-6084-471d-a778-7c5083700716" />

그림은 Nano Banana Pro에 의해 생성되었다.
