from ultralytics import YOLO

def main():
    model = YOLO('yolo26s.pt')  

    results = model.train(
        data='dataset_2/data_2.yaml',   
        epochs=100,              
        imgsz=640,                  
        batch=16,                     
        patience=20,                 
        device=0,                       # 0 = GPU
        project='runs/train',           # сохранение результатов
        name='kgo_detection_small26_half',    
        exist_ok=True,                   
        pretrained=True,                
        optimizer='auto',                
        seed=42,                        
    )

if __name__ == '__main__':
    main()