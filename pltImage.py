import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # 画像読み込みのためにインポート

# generated_images ディレクトリの内容をリスト表示
output_dir = 'generated_images'
if os.path.exists(output_dir):
    print(f"Directory '{output_dir}' contents:")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")

    # 保存された画像を読み込んで表示
    save_path = os.path.join(output_dir, 'sample_output.png')
    if os.path.exists(save_path):
        print(f"Loading and displaying {save_path}...")
        img = mpimg.imread(save_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title('Generated Image (from file)')
        plt.axis('off') # 軸を非表示にする
        plt.show()
    else:
        print(f"File '{save_path}' not found.")
else:
    print(f"Directory '{output_dir}' does not exist.")
