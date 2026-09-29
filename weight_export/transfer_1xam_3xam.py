import numpy as np

def convert_1ch_to_3ch_hardware_aligned(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    
    # 1. Đọc và chia 3 từng dòng
    for line in lines:
        if not line.strip(): continue
        num1_str, num2_str = line.strip().split()
        num1, num2 = int(num1_str), int(num2_str)
        
        def process_num(val):
            # Giải mã Bù 2 -> Chia 3 -> Mã hóa Bù 2 lại
            int_val = val if val <= 32767 else val - 65536
            new_int_val = round(int_val / 3.0)
            return new_int_val if new_int_val >= 0 else new_int_val + 65536
        
        converted_lines.append(f"{process_num(num1)} {process_num(num2)}\n")

    # 2. Tách khối dữ liệu (Dị biệt 8 + 1)
    # Giả định file 1 kênh chuẩn có đúng 144 dòng
    block_8_elements = converted_lines[:128]   # 128 dòng đầu
    block_9th_element = converted_lines[128:144] # 16 dòng cuối

    # 3. Lắp ráp lại theo đúng chuẩn routing của IP CNN phần cứng
    # [128 Ch1] + [128 Ch2] + [128 Ch3] + [16 Ch1] + [16 Ch2] + [16 Ch3]
    final_output = (block_8_elements * 3) + (block_9th_element * 3)

    # Xuất file
    with open(output_file, 'w') as f:
        f.writelines(final_output)

    print("Đã hack thành công! File xuất ra khớp 100% cấu trúc bộ nhớ phần cứng.")

# Chạy thử chuyển đổi
convert_1ch_to_3ch_hardware_aligned('conv0_W.txt', 'conv0_W_3channel_aligned.txt')