## soynlp를 이용한 sentence classification
- 이번 예시에서는 soynlp로 tokenize한 문서를 분류하는 문제를 풀어보도록 하겠습니다. 
- 대부분의 문서 분류에서는 tokenized된 단어를 기반으로 문서를 벡터화하여 분류기를 학습시킵니다.
- 즉, tokenizer의 성능에 따라 문서 분류기의 성능 또한 변할 수 있다는 것을 의미합니다. 

### 준비물
#### 1. 데이터
  - 분류기 학습을 위해 네이버 영화 리뷰 데이터를 다운받습니다(https://github.com/e9t/nsmc).
  - 200,000개 영화 리뷰 중 긍정적인 리뷰(label=1)와 부정적인 리뷰(label=2)를 분류하는 모델을 학습할 것입니다.
#### 2. Python library
  - tensorflow==2.1.0
  - pandas
  - soynlp
  
### 분류 모형
- 현재 두 가지 deep learning 기반의 분류기 예시가 준비되어 있습니다.
