from ultralytics import YOLO

# Load the YOLOv8n model, pre-trained on the COCO dataset
model = YOLO('yolov8n.pt')  # Using the YOLOv8 nano version

# Train the model using the specified dataset
training_outcome = model.train(data='/Users/sindhukavyaalahari/Documents/herts/finalproject/sindhu/data.yaml',
                               epochs=50,
                               imgsz=640)

# Debug: Display the state of the model's checkpoint
print("Checkpoint status:", model.ckpt)

# Save the trained model if the checkpoint is valid
if model.ckpt:
    model.save("yolov8_trained.pt")  # Save the model weights to a file
else:
    print("Error: No valid checkpoint found, model saving failed.")
