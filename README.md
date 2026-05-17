# CVio_Shrimp_Disease_Classification_Capstone_SU26

folder: onnx_models: chứa các models train pipeline shrimpxnet bình thường
folder: mobile_accuracy_optimization: chứa model yolo26n-cls train tới 31 epochs và model efficientb0 quantize các bit rate khác nhau

notebook:
- mobile-acc....: train 2 model và quantize (file chính cần chạy)
- tflite infere...: nhận model và inference
- training_and_log: train nhiều model (file phụ có thể cân nhắc chạy các model khác)
- onnx_to...: convert model từ onnx sang tflie