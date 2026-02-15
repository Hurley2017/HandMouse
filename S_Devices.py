import sounddevice as sd

print("--- AVAILABLE AUDIO DEVICES ---")
print(sd.query_devices())

print("\n--- INPUT DEVICES ONLY ---")
# Filter for input devices
for i, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        print(f"ID: {i} | Name: {device['name']}")