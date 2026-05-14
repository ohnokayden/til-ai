from cv_manager import CVManager

print("Loading AI Brain...")
# 1. This will trigger your __init__ and load best.pt
manager = CVManager()

print("Reading test image...")
# 2. Open your test image and convert it into raw bytes (just like your server would)
with open("helicopter+cruise.jpg", "rb") as image_file:
    image_bytes = image_file.read()

print("Running detection...")
# 3. Pass the bytes to your newly written cv function!
predictions = manager.cv(image_bytes)

print("\n--- Final Results ---")
# 4. Print out the formatted list of dictionaries
for item in predictions:
    print(item)