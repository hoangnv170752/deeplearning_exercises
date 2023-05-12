import numpy as np
from matplotlib import pyplot as plt

file_path = "../data/exercise1/full_numpy_bitmap_ambulance.npy"  # <= Các bạn thay từ bicycle bằng tên tương ứng của category các bạn chọn nhé
images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
print(images.shape)
train_images = images[:-10]  # Lấy tất cả ảnh, ngoại trừ 10 ảnh cuối ra làm bộ training.
test_images = images[-10:]  # Giữ 10 ảnh cuối làm bộ test

avg_image = np.average(train_images, axis=0) 
avg_image = np.reshape(avg_image, (28*28))
print(avg_image)
input(avg_image.shape)
# Bước 3: Các bạn sẽ visualize bức ảnh trung bình các bạn vửa tính được ở bước 2 bằng 2 dòng sau. Các bạn thử
# xem các bạn có nhận ra được category mà các bạn chọn bằng cách nhìn vào bức ảnh trung bình này không nhé
plt.imshow(avg_image)
plt.show()

# Bước 4: Các bạn chọn 1 index bất kì từ 0 đến 9. Ví dụ mình chọn index = 4
# Sau đó các bạn hãy tính tích vô hướng (dot product) của bức ảnh test này với bức ảnh trung bình các bạn tính được ở trên
index = 4  # Các bạn có thể thay đổi index tùy ý
test_image = test_images[index]
#TODO Các bạn tính tích vô hướng (dot product) của bức ảnh test và ảnh trung bình ở dòng dưới đây
# (các bạn có thể code trên nhiều hơn 1 dòng)
score = np.dot(avg_image, test_image)
print(score)

# Bước 5: Các bạn hãy lặp lại bước 1 đến 3 cho tất cả các categories còn lại (chú ý tại bước 1 các bạn không cần phân
# ra train với test images nữa nhé, coi như là dùng tất cả cho train). Sau đó các bạn hãy tính tích vô hướng của từng ảnh
# trung bình của ảnh test các bạn chọn ở bước 4 với từng bức ảnh trung bình này.
#
# Cuối cùng các bạn xem là liệu trong 10 score này, score tương ứng với tích vô hướng của ảnh test này với
# ảnh trung bình của category của chính nó có phải là score lớn nhất không nhé. Các bức ảnh trung bình mà các bạn tính ra
# có thể xem như là weight cho từng category mà các bạn vừa học ở bài 1 (tất nhiên là weight của mô hình sau khi đã
# train xong)

# Bước 6 (optional): Các bạn thử visualize 10 weight (avg_image) này trong cùng 1 ảnh kích thước 2x5 hoặc 5x2 để so sánh xem,
# weight của các categories nào dễ nhìn và weight nào không nhé
