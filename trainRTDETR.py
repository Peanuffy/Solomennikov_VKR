
from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt") 

    results = model.train(
        data='dataset_2/data_2.yaml',   
        epochs=100,                
        imgsz=640,                  
        batch=16,                   
        patience=20,                 
        device='cpu',                       # 0 = GPU
        project='runs/train',           # сохранение результатов
        name='kgo_detection_RTDETR',           
        exist_ok=True,                  
        pretrained=True,                
        optimizer='auto',                
        seed=42,                          
    )

if __name__ == '__main__':
    # Вызов функции main только при прямом запуске скрипта
    main()