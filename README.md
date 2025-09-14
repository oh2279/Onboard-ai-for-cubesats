# Onboard-ai-for-cubesats
소형 큐브 위성에 탑재될 구름 탐지를 위한 온보드 AI 구현 <br/>

실제 프로젝트에 쓰이는 폴더 : ./cloud_detection <br/>
backup/ : 백업용 가중치 <br/>
current/ : 현재 추론에 쓰이는 가중치 <br/>
inference_results/ : 추론 결과 <br/>
logs/ : 추론 log (처리한 이미지, 추론 시간 등) <br/>
raw_images/ : 처리할 이미지가 담긴 폴더 <br/>

inference.py : 추론 파일 <br/>
packaging.py : 지상 전송을 위해 처리된 이미지 및 geometry 데이터 등을 압축하는 폴더 (모델 ONNX 압축 여부는 구현중) <br/>

** 젯슨 전용 pytorch 라이브러리 (JetPack) 설치 필요

