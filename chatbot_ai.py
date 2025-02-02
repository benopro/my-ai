import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔹 Chọn mô hình AI (Có thể thay bằng LLaMA-2 hoặc mô hình nhẹ hơn nếu cần)
MODEL_NAME = "microsoft/phi-4"  # Hoặc "meta-llama/Llama-2-7b-chat-hf"
USE_GPU = torch.cuda.is_available()  # Kiểm tra xem có GPU không

# 🔹 Chọn thiết bị phù hợp
device = "cuda" if USE_GPU else "cpu"
dtype = torch.float16 if USE_GPU else torch.float32  # CPU không hỗ trợ float16

print(f"🚀 Đang tải mô hình {MODEL_NAME} trên {device.upper()}...")

# 🔹 Tải tokenizer và mô hình
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=dtype, 
    device_map=device
)

# 🔹 Hàm chạy chatbot
def chatbot(prompt):
    """Nhận đầu vào từ người dùng và sinh phản hồi từ AI."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 🔹 Chạy chatbot
if __name__ == "__main__":
    print("🤖 Chatbot AI - Nhập 'exit' để thoát")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == "exit":
            print("Tạm biệt!")
            break
        response = chatbot(user_input)
        print("AI:", response)

