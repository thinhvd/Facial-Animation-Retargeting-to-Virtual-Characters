import cv2

# Đọc ảnh
image = cv2.imread('./test_img.jpg')

# Tạo cửa sổ với tên 'ImageWindow' và kích thước 540x450
cv2.namedWindow('ImageWindow', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ImageWindow', 540, 450)

# Hiển thị ảnh trong cửa sổ đã tạo
cv2.imshow('ImageWindow', image)

# Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)

# Đóng tất cả các cửa sổ
cv2.destroyAllWindows()